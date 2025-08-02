"""
Wan Video Pipeline Implementation for DiffSynth Studio

This module implements the WanVideoPipeline, a comprehensive video generation system
developed by Alibaba (Wan AI). The pipeline supports multiple video generation modes:
- Text-to-Video (T2V): Generate videos from text prompts
- Image-to-Video (I2V): Generate videos from input images 
- Video-to-Video (V2V): Transform existing videos
- First-Last Frame to Video (FLF2V): Generate videos between two frames
- VACE: Video editing and completion
- VACE-E: Enhanced video editing with task-embodiment fusion for robot manipulation
- Various control mechanisms (camera, speed, depth, etc.)

Key Architecture Components:
1. BasePipeline: Base class providing common functionality
2. WanVideoPipeline: Main pipeline orchestrating all video generation tasks
3. PipelineUnit System: Modular processing units for different tasks
4. ModelConfig: Configuration management for loading models
5. Advanced Features: VRAM management, TeaCache, Unified Sequence Parallel
6. VACE-E Framework: Task and embodiment processing for robotic video generation

VACE-E Enhancements:
- Task Feature Processing: Text descriptions, dual-hand motions, object trajectories
- Dual-Hand Motion Processing: 20D format supporting left/right hand coordination
- Embodiment Processing: CLIP-encoded end-effector images
- Multi-modal Fusion: Cross-attention alignment between task and embodiment features
- Independent Weighted Fusion: Reduced correlation between modalities
- CLUB Loss Integration: Mutual information minimization for disentangled representations
- Two-Phase CLUB Training: Automatic q_θ(embodiment|task) approximation and MI minimization
- Backward Compatibility: Automatic conversion from legacy 10D single-hand format
- DiT Integration: Seamless integration with existing diffusion transformer architecture
"""

import torch, warnings, glob, os, types
import numpy as np
from PIL import Image
from einops import repeat, reduce
from typing import Optional, Union
from dataclasses import dataclass
from modelscope import snapshot_download
from einops import rearrange
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional
from typing_extensions import Literal

# Import DiffSynth components
from ..models import ModelManager, load_state_dict
from ..models.wan_video_dit import WanModel, RMSNorm, sinusoidal_embedding_1d
from ..models.wan_video_text_encoder import WanTextEncoder, T5RelativeEmbedding, T5LayerNorm
from ..models.wan_video_vae import WanVideoVAE, RMS_norm, CausalConv3d, Upsample
from ..models.wan_video_image_encoder import WanImageEncoder
from ..models.wan_video_vace import VaceWanModel
from ..models.wan_video_vace_E import VaceWanModel as VaceWanModelE, create_vace_model_from_dit
from ..models.wan_video_motion_controller import WanMotionControllerModel
from ..schedulers.flow_match import FlowMatchScheduler
from ..prompters import WanPrompter
from ..vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear, WanAutoCastLayerNorm
from ..lora import GeneralLoRALoader


# CLUB implementation for mutual information minimization
class CLUB(torch.nn.Module):
    """
    CLUB: Contrastive Log-ratio Upper Bound of Mutual Information
    
    This class provides the CLUB estimation to I(X,Y) for minimizing mutual information
    between task features and embodiment features in VACE-E training.
    
    Based on the original CLUB paper implementation for disentangled representation learning.
    """
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUB, self).__init__()
        # p_mu outputs mean of q(Y|X)
        self.p_mu = torch.nn.Sequential(torch.nn.Linear(x_dim, hidden_size//2),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(hidden_size//2, y_dim))
        # p_logvar outputs log of variance of q(Y|X)
        self.p_logvar = torch.nn.Sequential(torch.nn.Linear(x_dim, hidden_size//2),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(hidden_size//2, y_dim),
                                       torch.nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
    
    def forward(self, x_samples, y_samples): 
        mu, logvar = self.get_mu_logvar(x_samples)
        
        # log of conditional probability of positive sample pairs
        positive = - (mu - y_samples)**2 /2./logvar.exp()  
        
        prediction_1 = mu.unsqueeze(1)          # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)    # shape [1,nsample,dim]

        # log of conditional probability of negative sample pairs
        negative = - ((y_samples_1 - prediction_1)**2).mean(dim=1)/2./logvar.exp() 

        return (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()

    def loglikeli(self, x_samples, y_samples): # unnormalized loglikelihood 
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)
    
    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)


class BasePipeline(torch.nn.Module):
    """
    Base pipeline class providing fundamental functionality for all diffusion pipelines.
    
    This class serves as the foundation for all video/image generation pipelines in DiffSynth,
    providing common utilities for device management, shape checking, preprocessing, 
    VRAM management, and other core functionalities.
    
    Key Features:
    - Device and dtype management for efficient computation
    - Shape validation and automatic resizing
    - Image/video preprocessing utilities
    - VRAM management for large models
    - Noise generation for diffusion processes
    """

    def __init__(
        self,
        device="cuda", torch_dtype=torch.float16,
        height_division_factor=64, width_division_factor=64,
        time_division_factor=None, time_division_remainder=None,
    ):
        """
        Initialize the base pipeline.
        
        Args:
            device: Computation device ('cuda' or 'cpu')
            torch_dtype: Data type for intermediate variables (not models)
            height_division_factor: Height must be divisible by this factor
            width_division_factor: Width must be divisible by this factor  
            time_division_factor: Number of frames must follow this pattern
            time_division_remainder: Remainder for frame count validation
        """
        super().__init__()
        # The device and torch_dtype is used for the storage of intermediate variables, not models.
        self.device = device
        self.torch_dtype = torch_dtype
        # The following parameters are used for shape check.
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.vram_management_enabled = False
        
        
    def to(self, *args, **kwargs):
        """Move pipeline to specified device/dtype while updating internal state."""
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.device = device
        if dtype is not None:
            self.torch_dtype = dtype
        super().to(*args, **kwargs)
        return self


    def check_resize_height_width(self, height, width, num_frames=None):
        """
        Validate and adjust dimensions to meet model requirements.
        
        Video generation models typically require specific dimension constraints:
        - Height/width must be divisible by certain factors (usually 8, 16, or 64)
        - Frame count must follow specific patterns for temporal consistency
        
        Args:
            height: Desired video height
            width: Desired video width  
            num_frames: Number of frames (optional)
            
        Returns:
            Adjusted (height, width) or (height, width, num_frames)
        """
        # Shape check
        if height % self.height_division_factor != 0:
            height = (height + self.height_division_factor - 1) // self.height_division_factor * self.height_division_factor
            print(f"height % {self.height_division_factor} != 0. We round it up to {height}.")
        if width % self.width_division_factor != 0:
            width = (width + self.width_division_factor - 1) // self.width_division_factor * self.width_division_factor
            print(f"width % {self.width_division_factor} != 0. We round it up to {width}.")
        if num_frames is None:
            return height, width
        else:
            if num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames = (num_frames + self.time_division_factor - 1) // self.time_division_factor * self.time_division_factor + self.time_division_remainder
                print(f"num_frames % {self.time_division_factor} != {self.time_division_remainder}. We round it up to {num_frames}.")
            return height, width, num_frames


    def preprocess_image(self, image, torch_dtype=None, device=None, pattern="B C H W", min_value=-1, max_value=1):
        """
        Convert PIL Image to torch tensor with specified format and value range.
        
        Args:
            image: PIL Image to convert
            torch_dtype: Target tensor dtype
            device: Target device
            pattern: Tensor dimension pattern (e.g., "B C H W" for batch, channel, height, width)
            min_value: Minimum pixel value after normalization
            max_value: Maximum pixel value after normalization
            
        Returns:
            Preprocessed torch tensor
        """
        # Transform a PIL.Image to torch.Tensor
        image = torch.Tensor(np.array(image, dtype=np.float32))
        image = image.to(dtype=torch_dtype or self.torch_dtype, device=device or self.device)
        image = image * ((max_value - min_value) / 255) + min_value
        image = repeat(image, f"H W C -> {pattern}", **({"B": 1} if "B" in pattern else {}))
        return image


    def preprocess_video(self, video, torch_dtype=None, device=None, pattern="B C T H W", min_value=-1, max_value=1):
        """
        Convert list of PIL Images to video tensor.
        
        Args:
            video: List of PIL Images representing video frames
            torch_dtype: Target tensor dtype
            device: Target device  
            pattern: Tensor dimension pattern (e.g., "B C T H W" for batch, channel, time, height, width)
            min_value: Minimum pixel value after normalization
            max_value: Maximum pixel value after normalization
            
        Returns:
            Video tensor with shape according to pattern
        """
        # Transform a list of PIL.Image to torch.Tensor
        video = [self.preprocess_image(image, torch_dtype=torch_dtype, device=device, min_value=min_value, max_value=max_value) for image in video]
        video = torch.stack(video, dim=pattern.index("T") // 2)
        return video


    def vae_output_to_image(self, vae_output, pattern="B C H W", min_value=-1, max_value=1):
        """
        Convert VAE latent tensor back to PIL Image.
        
        Args:
            vae_output: VAE decoder output tensor
            pattern: Input tensor dimension pattern
            min_value: Minimum value in tensor
            max_value: Maximum value in tensor
            
        Returns:
            PIL Image
        """
        # Transform a torch.Tensor to PIL.Image
        if pattern != "H W C":
            vae_output = reduce(vae_output, f"{pattern} -> H W C", reduction="mean")
        image = ((vae_output - min_value) * (255 / (max_value - min_value))).clip(0, 255)
        image = image.to(device="cpu", dtype=torch.uint8)
        image = Image.fromarray(image.numpy())
        return image


    def vae_output_to_video(self, vae_output, pattern="B C T H W", min_value=-1, max_value=1):
        """
        Convert VAE latent tensor back to list of PIL Images (video).
        
        Args:
            vae_output: VAE decoder output tensor
            pattern: Input tensor dimension pattern  
            min_value: Minimum value in tensor
            max_value: Maximum value in tensor
            
        Returns:
            List of PIL Images representing video frames
        """
        # Transform a torch.Tensor to list of PIL.Image
        if pattern != "T H W C":
            vae_output = reduce(vae_output, f"{pattern} -> T H W C", reduction="mean")
        video = [self.vae_output_to_image(image, pattern="H W C", min_value=min_value, max_value=max_value) for image in vae_output]
        return video


    def load_models_to_device(self, model_names=[]):
        """
        Manage model placement for VRAM optimization.
        
        When VRAM management is enabled, this method:
        1. Offloads unused models to CPU/storage
        2. Loads required models to GPU
        3. Clears VRAM cache for efficiency
        
        Args:
            model_names: List of model names to load to device
        """
        if self.vram_management_enabled:
            # offload models
            for name, model in self.named_children():
                if name not in model_names:
                    if hasattr(model, "vram_management_enabled") and model.vram_management_enabled:
                        for module in model.modules():
                            if hasattr(module, "offload"):
                                module.offload()
                    else:
                        model.cpu()
            torch.cuda.empty_cache()
            # onload models
            for name, model in self.named_children():
                if name in model_names:
                    if hasattr(model, "vram_management_enabled") and model.vram_management_enabled:
                        for module in model.modules():
                            if hasattr(module, "onload"):
                                module.onload()
                    else:
                        model.to(self.device)


    def generate_noise(self, shape, seed=None, rand_device="cpu", rand_torch_dtype=torch.float32, device=None, torch_dtype=None):
        """
        Generate Gaussian noise for diffusion process.
        
        Args:
            shape: Tensor shape for noise
            seed: Random seed for reproducibility  
            rand_device: Device for random number generation
            rand_torch_dtype: Data type for random generation
            device: Target device for final tensor
            torch_dtype: Target dtype for final tensor
            
        Returns:
            Random noise tensor
        """
        # Initialize Gaussian noise
        generator = None if seed is None else torch.Generator(rand_device).manual_seed(seed)
        noise = torch.randn(shape, generator=generator, device=rand_device, dtype=rand_torch_dtype)
        noise = noise.to(dtype=torch_dtype or self.torch_dtype, device=device or self.device)
        return noise


    def enable_cpu_offload(self):
        """Deprecated method. Use enable_vram_management instead."""
        warnings.warn("`enable_cpu_offload` will be deprecated. Please use `enable_vram_management`.")
        self.vram_management_enabled = True
        
        
    def get_vram(self):
        """Get total VRAM available on current device in GB."""
        return torch.cuda.mem_get_info(self.device)[1] / (1024 ** 3)
    
    
    def freeze_except(self, model_names):
        """
        Freeze all models except specified ones for training.
        
        Args:
            model_names: List of model names to keep trainable
        """
        for name, model in self.named_children():
            if name in model_names:
                model.train()
                model.requires_grad_(True)
            else:
                model.eval()
                model.requires_grad_(False)


@dataclass
class ModelConfig:
    """
    Configuration for loading models from various sources.
    
    This class provides flexible model loading from:
    - Local file paths
    - ModelScope/HuggingFace repositories
    - Multiple file patterns for large models
    
    Attributes:
        path: Local file path(s) to model
        model_id: Repository ID for remote download
        origin_file_pattern: File pattern to download from repository
        download_resource: Source platform ("ModelScope" or "HuggingFace")
        offload_device: Device for initial model storage
        offload_dtype: Data type for model storage
    """
    path: Union[str, list[str]] = None
    model_id: str = None
    origin_file_pattern: Union[str, list[str]] = None
    download_resource: str = "ModelScope"
    offload_device: Optional[Union[str, torch.device]] = None
    offload_dtype: Optional[torch.dtype] = None
    skip_download: bool = False

    def download_if_necessary(self, local_model_path="./models", skip_download=False, use_usp=False):
        """
        Download model files if not available locally.
        
        Handles distributed downloading when using Unified Sequence Parallel (USP):
        - Only rank 0 downloads files
        - Other ranks wait for completion
        - Synchronization via distributed barriers
        
        Args:
            local_model_path: Base directory for model storage
            skip_download: Skip downloading if files exist
            use_usp: Whether using Unified Sequence Parallel
        """
        if self.path is None:
            # Check model_id and origin_file_pattern
            if self.model_id is None:
                raise ValueError(f"""No valid model files. Please use `ModelConfig(path="xxx")` or `ModelConfig(model_id="xxx/yyy", origin_file_pattern="zzz")`.""")
            
            # Skip if not in rank 0
            if use_usp:
                import torch.distributed as dist
                skip_download = dist.get_rank() != 0
                
            # Check whether the origin path is a folder
            if self.origin_file_pattern is None or self.origin_file_pattern == "":
                self.origin_file_pattern = ""
                allow_file_pattern = None
                is_folder = True
            elif isinstance(self.origin_file_pattern, str) and self.origin_file_pattern.endswith("/"):
                allow_file_pattern = self.origin_file_pattern + "*"
                is_folder = True
            else:
                allow_file_pattern = self.origin_file_pattern
                is_folder = False
            
            # Download
            skip_download = skip_download or self.skip_download
            if not skip_download:
                downloaded_files = glob.glob(self.origin_file_pattern, root_dir=os.path.join(local_model_path, self.model_id))
                snapshot_download(
                    self.model_id,
                    local_dir=os.path.join(local_model_path, self.model_id),
                    allow_file_pattern=allow_file_pattern,
                    ignore_file_pattern=downloaded_files,
                    local_files_only=False
                )
            
            # Let rank 1, 2, ... wait for rank 0
            if use_usp:
                import torch.distributed as dist
                dist.barrier(device_ids=[dist.get_rank()])
                
            # Return downloaded files
            if is_folder:
                self.path = os.path.join(local_model_path, self.model_id, self.origin_file_pattern)
            else:
                self.path = glob.glob(os.path.join(local_model_path, self.model_id, self.origin_file_pattern))
            if isinstance(self.path, list) and len(self.path) == 1:
                self.path = self.path[0]


class WanVideoPipeline(BasePipeline):
    """
    Main Wan Video Generation Pipeline.
    
    This is the central class for all Wan video generation tasks. It orchestrates
    multiple neural network components to generate high-quality videos from various inputs.
    
    Architecture Overview:
    - Text Encoder: Processes text prompts into embeddings
    - Image Encoder: Handles input images for I2V tasks  
    - DiT (Diffusion Transformer): Core denoising model
    - VAE: Encodes/decodes between pixel and latent space
    - Motion Controller: Controls video motion dynamics
    - VACE: Video editing and completion capabilities
    - Various Control Units: Modular processing components
    
    Supported Generation Modes:
    1. Text-to-Video (T2V): Generate videos from text descriptions
    2. Image-to-Video (I2V): Generate videos from input images
    3. Video-to-Video (V2V): Transform existing videos  
    4. First-Last Frame: Generate between two frames
    5. VACE: Video editing with masks and reference images
    6. Control modes: Camera control, speed control, depth control
    """

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16, tokenizer_path=None):
        """
        Initialize the Wan Video Pipeline.
        
        Args:
            device: Computation device 
            torch_dtype: Data type for computations (bfloat16 recommended)
            tokenizer_path: Path to tokenizer files
        """
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16, time_division_factor=4, time_division_remainder=1
        )
        
        # Initialize core components
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        
        # Model components (initialized later via from_pretrained)
        self.text_encoder: WanTextEncoder = None      # T5-based text encoder
        self.image_encoder: WanImageEncoder = None    # CLIP-based image encoder  
        self.dit: WanModel = None                     # Diffusion Transformer
        self.vae: WanVideoVAE = None                  # Video VAE for latent encoding
        self.motion_controller: WanMotionControllerModel = None  # Motion dynamics
        self.vace: VaceWanModel = None                # Video editing model
        self.vace_e: VaceWanModelE = None             # Enhanced video editing with task-embodiment fusion
        
        # Models that run during each diffusion step
        self.in_iteration_models = ("dit", "motion_controller", "vace", "vace_e")
        
        # Processing pipeline: modular units for different tasks
        self.unit_runner = PipelineUnitRunner()
        self.units = [
            WanVideoUnit_ShapeChecker(),           # Validate input dimensions
            WanVideoUnit_NoiseInitializer(),       # Initialize noise tensor
            WanVideoUnit_InputVideoEmbedder(),     # Encode input videos
            WanVideoUnit_PromptEmbedder(),         # Encode text prompts
            WanVideoUnit_ImageEmbedder(),          # Encode input images
            WanVideoUnit_FunControl(),             # Handle control videos
            WanVideoUnit_FunReference(),           # Handle reference images
            WanVideoUnit_FunCameraControl(),       # Camera movement control
            WanVideoUnit_SpeedControl(),           # Motion speed control
            WanVideoUnit_VACE(),                   # Video editing features
            WanVideoUnit_VACE_E(),                 # Enhanced video editing with task-embodiment fusion
            WanVideoUnit_UnifiedSequenceParallel(), # Distributed computing
            WanVideoUnit_TeaCache(),               # Attention caching optimization
            WanVideoUnit_CfgMerger(),              # Classifier-free guidance merging
        ]
        
        # Core model function for diffusion process
        self.model_fn = model_fn_wan_video
        
        # CLUB estimator for mutual information minimization between task and embodiment features
        # Initialized with default dimensions, will be properly configured when models are loaded
        self.club_estimator = None
        self.club_optimizer = None  # Separate optimizer for CLUB estimator
        self.club_lambda = 1.0  # Weight for CLUB loss
        self.club_update_freq = 1  # Update CLUB estimator every N training steps
        self.club_training_steps = 5  # Number of CLUB training steps per main training step
        self.enable_club_loss = True  # Whether CLUB loss is enabled
        
    
    def load_lora(self, module, path, alpha=1):
        """
        Load LoRA (Low-Rank Adaptation) weights for fine-tuning.
        
        LoRA allows efficient fine-tuning by adding low-rank matrices to existing layers.
        This enables style transfer, concept learning, and domain adaptation without
        retraining the entire model.
        
        Args:
            module: Target model module (e.g., self.dit, self.vace)
            path: Path to LoRA weight file (.safetensors or .pth)
            alpha: LoRA scaling factor
        """
        loader = GeneralLoRALoader(torch_dtype=self.torch_dtype, device=self.device)
        lora = load_state_dict(path, torch_dtype=self.torch_dtype, device=self.device)
        loader.load(module, lora, alpha=alpha)

        
    def training_loss(self, training_step=0, **inputs):
        """
        Compute training loss for model fine-tuning with CLUB-based mutual information minimization.
        
        Implements the flow matching training objective with disentangled representation learning:
        1. Sample random timestep
        2. Add noise to clean data  
        3. Predict the noise/flow direction with task and embodiment features
        4. Compute MSE loss with proper weighting
        5. Add CLUB loss to minimize mutual information between task and embodiment features
        
        Args:
            training_step: Current training step (for CLUB estimator updates)
            return_detailed_losses: If True, return dict with loss breakdown; if False, return scalar total_loss
            **inputs: Training inputs including latents, noise, prompts, VACE-E features, etc.
            
        Returns:
            If return_detailed_losses=True: Dict containing:
                - 'total_loss': Combined loss for backpropagation
                - 'flow_loss': Flow matching loss component  
                - 'club_loss': CLUB mutual information loss component
            If return_detailed_losses=False: Scalar tensor (total_loss) for training framework
        """
        timestep_id = torch.randint(0, self.scheduler.num_train_timesteps, (1,))
        timestep = self.scheduler.timesteps[timestep_id].to(dtype=self.torch_dtype, device=self.device)
        
        inputs["latents"] = self.scheduler.add_noise(inputs["input_latents"], inputs["noise"], timestep)
        training_target = self.scheduler.training_target(inputs["input_latents"], inputs["noise"], timestep)
        
        # Initialize return_intermediate_features flag for VACE-E
        # Check if VACE-E is enabled and we have the necessary features
        has_vace_e_features = (
            inputs.get("vace_e_context") is not None and 
            self.vace_e is not None and
            (inputs["vace_e_context"].get("text_features") is not None or
             inputs["vace_e_context"].get("hand_motion_sequence") is not None or
             inputs["vace_e_context"].get("object_trajectory_sequence") is not None or
             inputs["vace_e_context"].get("embodiment_image_features") is not None)
        )
        
        # Temporarily modify VACE-E context to request intermediate features if available
        original_vace_e_context = None
        if has_vace_e_features:
            original_vace_e_context = inputs["vace_e_context"].copy()
            inputs["vace_e_context"]["return_intermediate_features"] = True
        
        # Get model prediction (potentially with intermediate features)
        noise_pred = self.model_fn(**inputs, timestep=timestep)
        
        # Restore original context
        if has_vace_e_features:
            inputs["vace_e_context"] = original_vace_e_context
        
        # Compute flow matching loss
        flow_loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        flow_loss = flow_loss * self.scheduler.training_weight(timestep)
        
        # Initialize CLUB loss
        club_loss = torch.tensor(0.0, device=self.device, dtype=self.torch_dtype)
        
        # Compute CLUB loss if VACE-E features are available and CLUB loss is enabled
        if (self.enable_club_loss and 
            has_vace_e_features and 
            '_current_task_features' in globals() and 
            '_current_embodiment_features' in globals()):
            
            task_features = globals()['_current_task_features']
            embodiment_features = globals()['_current_embodiment_features']
            
            if task_features is not None and embodiment_features is not None:
                # Reduce feature dimensions to prevent memory issues
                print(f"Original feature shapes: task={task_features.shape}, embodiment={embodiment_features.shape}")
                
                # Use global average pooling to reduce sequence dimension if needed
                if task_features.dim() == 3:  # [batch, seq_len, dim]
                    task_reduced = torch.mean(task_features, dim=1)  # [batch, dim]
                else:
                    task_reduced = task_features
                    
                if embodiment_features.dim() == 3:  # [batch, seq_len, dim] 
                    embodiment_reduced = torch.mean(embodiment_features, dim=1)  # [batch, dim]
                else:
                    embodiment_reduced = embodiment_features
                
                # # Further reduce dimensions if still too large (limit to max 512 per feature)
                # max_dim = 512
                # if task_reduced.shape[-1] > max_dim:
                #     # Simple linear projection to reduce dimension
                #     if not hasattr(self, 'task_dim_reducer'):
                #         self.task_dim_reducer = torch.nn.Linear(task_reduced.shape[-1], max_dim).to(device=self.device, dtype=self.torch_dtype)
                #         print(f"Created task dimension reducer: {task_reduced.shape[-1]} -> {max_dim}")
                #     task_reduced = self.task_dim_reducer(task_reduced)
                    
                # if embodiment_reduced.shape[-1] > max_dim:
                #     # Simple linear projection to reduce dimension
                #     if not hasattr(self, 'embodiment_dim_reducer'):
                #         self.embodiment_dim_reducer = torch.nn.Linear(embodiment_reduced.shape[-1], max_dim).to(device=self.device, dtype=self.torch_dtype)
                #         print(f"Created embodiment dimension reducer: {embodiment_reduced.shape[-1]} -> {max_dim}")
                #     embodiment_reduced = self.embodiment_dim_reducer(embodiment_reduced)
                
                # print(f"Reduced feature shapes: task={task_reduced.shape}, embodiment={embodiment_reduced.shape}")
                
                # Initialize CLUB estimator if not already done
                if self.club_estimator is None:
                    # Get dimensions from reduced features
                    task_dim = task_reduced.shape[-1]
                    embodiment_dim = embodiment_reduced.shape[-1]
                    hidden_size = min(256, max(task_dim, embodiment_dim))  # Conservative hidden size
                    
                    print(f"CLUB dimensions: task_dim={task_dim}, embodiment_dim={embodiment_dim}, hidden_size={hidden_size}")
                    self.club_estimator = CLUB(task_dim, embodiment_dim, hidden_size).to(device=self.device, dtype=self.torch_dtype)
                    # Initialize CLUB optimizer
                    self.club_optimizer = torch.optim.Adam(self.club_estimator.parameters(), lr=getattr(self, 'club_lr', 1e-3))
                    print(f"🎯 Initialized CLUB estimator with dtype {self.torch_dtype} successfully!")
                
                # Use reduced features for CLUB computation (already 2D)
                # Ensure features have the correct dtype for CLUB computation
                task_flat = task_reduced.to(dtype=self.torch_dtype)
                embodiment_flat = embodiment_reduced.to(dtype=self.torch_dtype)
                print(f"Final features for CLUB: task_flat={task_flat.shape} {task_flat.dtype}, embodiment_flat={embodiment_flat.shape} {embodiment_flat.dtype}")
                
                # Phase 1: Train CLUB estimator to approximate q_θ(embodiment|task)
                if training_step % self.club_update_freq == 0:
                    club_training_loss = self.train_club_estimator(task_flat, embodiment_flat)
                    # Note: CLUB training loss is for monitoring only, not included in main loss
                
                # Phase 2: Compute MI upper bound using trained CLUB estimator
                with torch.no_grad():
                    self.club_estimator.eval()
                    mi_upper_bound = self.club_estimator(task_flat, embodiment_flat)
                    self.club_estimator.train()
                
                # Add MI upper bound to loss (we want to minimize this)
                club_loss += self.club_lambda * mi_upper_bound
                
                # Clean up global variables
                if '_current_task_features' in globals():
                    del globals()['_current_task_features']
                if '_current_embodiment_features' in globals():
                    del globals()['_current_embodiment_features']
        
        # Combine losses
        total_loss = flow_loss + club_loss

        log_loss = {
                'total_loss': total_loss,
                'flow_loss': flow_loss,
                'club_loss': club_loss
            }
        
        return total_loss, log_loss

    def configure_club_loss(self, lambda_weight=1.0, update_freq=1, training_steps=5, club_lr=1e-3, enable=True):
        """
        Configure CLUB loss parameters for mutual information minimization.
        
        Args:
            lambda_weight: Weight for CLUB loss in total loss (default: 1.0)
            update_freq: Update CLUB estimator every N training steps (default: 1)
            training_steps: Number of CLUB training steps per update (default: 5)
            club_lr: Learning rate for CLUB optimizer (default: 1e-3)
            enable: Whether to enable CLUB loss computation (default: True)
        """
        self.club_lambda = lambda_weight
        self.club_update_freq = update_freq
        self.club_training_steps = training_steps
        self.club_lr = club_lr
        self.enable_club_loss = enable
        
        print(f"🎯 CLUB loss configured: lambda={lambda_weight}, update_freq={update_freq}, training_steps={training_steps}, lr={club_lr}, enabled={enable}")
    
    def train_club_estimator(self, task_features, embodiment_features):
        """
        Train the CLUB estimator to approximate q_θ(embodiment|task).
        
        This method trains the CLUB estimator using the learning_loss to maximize
        the log-likelihood of the conditional distribution approximation.
        
        Args:
            task_features: Task feature tensor [batch, task_dim]
            embodiment_features: Embodiment feature tensor [batch, embodiment_dim]
        
        Returns:
            Average training loss over all training steps
        """
        if self.club_estimator is None:
            return 0.0
        
        self.club_estimator.train()
        total_loss = 0.0
        
        for step in range(self.club_training_steps):
            # Train CLUB estimator to approximate q_θ(embodiment|task)
            club_loss = self.club_estimator.learning_loss(task_features, embodiment_features)
            
            # Update CLUB estimator parameters
            self.club_optimizer.zero_grad()
            club_loss.backward(retain_graph=True)  # Retain graph for potential multiple uses
            self.club_optimizer.step()
            
            total_loss += club_loss.item()
        
        self.club_estimator.eval()
        return total_loss / self.club_training_steps
    
    def get_club_estimator_info(self):
        """
        Get information about the current CLUB estimator.
        
        Returns:
            Dict with CLUB estimator information
        """
        if self.club_estimator is None:
            return {"initialized": False}
        
        # Get parameter count
        total_params = sum(p.numel() for p in self.club_estimator.parameters())
        trainable_params = sum(p.numel() for p in self.club_estimator.parameters() if p.requires_grad)
        
        return {
            "initialized": True,
            "lambda_weight": self.club_lambda,
            "update_freq": self.club_update_freq,
            "training_steps": self.club_training_steps,
            "club_lr": getattr(self, 'club_lr', 'not_set'),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": next(self.club_estimator.parameters()).device,
            "optimizer_initialized": self.club_optimizer is not None,
        }


    def enable_vram_management(self, num_persistent_param_in_dit=None, vram_limit=None, vram_buffer=0.5):
        """
        Enable intelligent VRAM management for large models.
        
        This system dynamically offloads/onloads model components based on usage,
        enabling larger models to run on limited VRAM by moving unused parts to CPU.
        
        Strategies:
        1. Parameter-based: Keep only N parameters of DiT in VRAM
        2. Limit-based: Maintain VRAM usage below specified threshold
        3. Automatic offloading: Move unused modules to CPU storage
        
        Args:
            num_persistent_param_in_dit: Number of DiT parameters to keep in VRAM
            vram_limit: Maximum VRAM usage in GB (auto-detected if None)
            vram_buffer: Safety buffer in GB
        """
        self.vram_management_enabled = True
        if num_persistent_param_in_dit is not None:
            vram_limit = None
        else:
            if vram_limit is None:
                vram_limit = self.get_vram()
            vram_limit = vram_limit - vram_buffer
            
        # Configure VRAM management for each model component
        if self.text_encoder is not None:
            dtype = next(iter(self.text_encoder.parameters())).dtype
            enable_vram_management(
                self.text_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Embedding: AutoWrappedModule,
                    T5RelativeEmbedding: AutoWrappedModule,
                    T5LayerNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.dit is not None:
            dtype = next(iter(self.dit.parameters())).dtype
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.dit,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: WanAutoCastLayerNorm,
                    RMSNorm: AutoWrappedModule,
                    torch.nn.Conv2d: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                max_num_param=num_persistent_param_in_dit,
                overflow_module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.vae is not None:
            dtype = next(iter(self.vae.parameters())).dtype
            enable_vram_management(
                self.vae,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    RMS_norm: AutoWrappedModule,
                    CausalConv3d: AutoWrappedModule,
                    Upsample: AutoWrappedModule,
                    torch.nn.SiLU: AutoWrappedModule,
                    torch.nn.Dropout: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=self.device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
            )
        if self.image_encoder is not None:
            dtype = next(iter(self.image_encoder.parameters())).dtype
            enable_vram_management(
                self.image_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        if self.motion_controller is not None:
            dtype = next(iter(self.motion_controller.parameters())).dtype
            enable_vram_management(
                self.motion_controller,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        if self.vace is not None:
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.vace,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                    RMSNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.vace_e is not None:
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.vace_e,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                    RMSNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
            
            
    def initialize_usp(self):
        """
        Initialize Unified Sequence Parallel (USP) for distributed computing.
        
        USP enables efficient distributed training/inference by:
        1. Splitting sequences across multiple GPUs
        2. Using ring-based communication patterns
        3. Optimizing memory usage through sequence parallelism
        
        This is particularly useful for long video generation tasks.
        """
        import torch.distributed as dist
        from xfuser.core.distributed import initialize_model_parallel, init_distributed_environment
        dist.init_process_group(backend="nccl", init_method="env://")
        init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=1,
            ulysses_degree=dist.get_world_size(),
        )
        torch.cuda.set_device(dist.get_rank())
            
            
    def enable_usp(self):
        """
        Enable Unified Sequence Parallel by patching attention mechanisms.
        
        Replaces standard attention forward functions with USP-aware versions
        that can split sequences across multiple devices efficiently.
        """
        from xfuser.core.distributed import get_sequence_parallel_world_size
        from ..distributed.xdit_context_parallel import usp_attn_forward, usp_dit_forward

        for block in self.dit.blocks:
            block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
        self.dit.forward = types.MethodType(usp_dit_forward, self.dit)
        self.sp_size = get_sequence_parallel_world_size()
        self.use_unified_sequence_parallel = True


    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/*"),
        local_model_path: str = "./models",
        skip_download: bool = False,
        redirect_common_files: bool = True,
        use_usp=False,
        # VACE-E configuration
        enable_vace_e: bool = True,
        vace_e_layers: tuple = (0, 5, 10, 15, 20, 25),
        vace_e_task_processing: bool = True,
    ):
        """
        Create WanVideoPipeline from pretrained models.
        
        This factory method handles:
        1. Model downloading from repositories
        2. Component initialization and loading
        3. Tokenizer setup
        4. Optional USP configuration
        5. VACE-E initialization with DiT weights
        
        Args:
            torch_dtype: Computation data type
            device: Target device
            model_configs: List of model configurations to load
            tokenizer_config: Tokenizer configuration
            local_model_path: Local storage path for models
            skip_download: Skip downloading if files exist
            redirect_common_files: Reuse common files across models
            use_usp: Enable Unified Sequence Parallel
            
            # VACE-E Enhanced Video Editing Configuration
            enable_vace_e: Whether to enable VACE-E support (default: True)
            vace_e_layers: DiT layer indices for VACE-E hints (default: (0, 5, 10, 15, 20, 25))
            vace_e_task_processing: Enable task feature processing (text, motion, trajectory) (default: True)
            
        Returns:
            WanVideoPipeline: Initialized pipeline with all components loaded
            
        Note:
            VACE-E model is created using create_vace_model_from_dit() and initialized
            with pretrained DiT weights since no pretrained VACE-E model exists yet.
        """
        # Redirect model path
        if redirect_common_files:
            redirect_dict = {
                "models_t5_umt5-xxl-enc-bf16.pth": "Wan-AI/Wan2.1-T2V-1.3B",
                "Wan2.1_VAE.pth": "Wan-AI/Wan2.1-T2V-1.3B",
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": "Wan-AI/Wan2.1-I2V-14B-480P",
            }
            for model_config in model_configs:
                if model_config.origin_file_pattern is None or model_config.model_id is None:
                    continue
                if model_config.origin_file_pattern in redirect_dict and model_config.model_id != redirect_dict[model_config.origin_file_pattern]:
                    print(f"To avoid repeatedly downloading model files, ({model_config.model_id}, {model_config.origin_file_pattern}) is redirected to ({redirect_dict[model_config.origin_file_pattern]}, {model_config.origin_file_pattern}). You can use `redirect_common_files=False` to disable file redirection.")
                    model_config.model_id = redirect_dict[model_config.origin_file_pattern]
        
        # Initialize pipeline
        pipe = WanVideoPipeline(device=device, torch_dtype=torch_dtype)
        if use_usp: pipe.initialize_usp()
        
        # Download and load models
        model_manager = ModelManager()
        for model_config in model_configs:
            model_config.download_if_necessary(local_model_path, skip_download=skip_download, use_usp=use_usp)
            model_manager.load_model(
                model_config.path,
                device=model_config.offload_device or device,
                torch_dtype=model_config.offload_dtype or torch_dtype
            )
        
        # Load models
        pipe.text_encoder = model_manager.fetch_model("wan_video_text_encoder")
        pipe.dit = model_manager.fetch_model("wan_video_dit")
        pipe.vae = model_manager.fetch_model("wan_video_vae")
        pipe.image_encoder = model_manager.fetch_model("wan_video_image_encoder")
        pipe.motion_controller = model_manager.fetch_model("wan_video_motion_controller")
        pipe.vace = model_manager.fetch_model("wan_video_vace")
        # pipe.vace_e = model_manager.fetch_model("wan_video_vace_e")
        
        # Initialize VACE-E model using DiT weights (no pre-trained VACE-E model available yet)
        if pipe.dit is not None and enable_vace_e:
            try:
                pipe.vace_e = create_vace_model_from_dit(
                    pipe.dit,
                    vace_layers=vace_e_layers,
                    enable_task_processing=vace_e_task_processing
                )
                # Ensure VACE-E model is on the correct device
                pipe.vace_e = pipe.vace_e.to(device=device, dtype=torch_dtype)
                print(f"✅ VACE-E model initialized with DiT weights")
                print(f"   VACE-E layers: {vace_e_layers}")
                print(f"   Task processing: {vace_e_task_processing}")
                print(f"   Device: {device}, Dtype: {torch_dtype}")
            except Exception as e:
                print(f"⚠️ VACE-E initialization failed: {e}")
                print("Continuing without VACE-E support...")
                pipe.vace_e = None
        else:
            pipe.vace_e = None
            if not enable_vace_e:
                print("ℹ️ VACE-E disabled by user configuration")

        # Initialize tokenizer
        tokenizer_config.download_if_necessary(local_model_path, skip_download=skip_download)
        pipe.prompter.fetch_models(pipe.text_encoder)
        pipe.prompter.fetch_tokenizer(tokenizer_config.path)
        
        # Unified Sequence Parallel
        if use_usp: pipe.enable_usp()
        return pipe


    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: Optional[str] = "",
        # Image-to-video
        input_image: Optional[Image.Image] = None,
        # First-last-frame-to-video
        end_image: Optional[Image.Image] = None,
        # Video-to-video
        input_video: Optional[list[Image.Image]] = None,
        denoising_strength: Optional[float] = 1.0,
        # ControlNet
        control_video: Optional[list[Image.Image]] = None,
        reference_image: Optional[Image.Image] = None,
        # Camera control
        camera_control_direction: Optional[Literal["Left", "Right", "Up", "Down", "LeftUp", "LeftDown", "RightUp", "RightDown"]] = None,
        camera_control_speed: Optional[float] = 1/54,
        camera_control_origin: Optional[tuple] = (0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0),
        # VACE
        vace_video: Optional[list[Image.Image]] = None,
        vace_video_mask: Optional[Image.Image] = None,
        vace_reference_image: Optional[Image.Image] = None,
        vace_scale: Optional[float] = 1.0,
        # VACE-E: Enhanced video editing with task-embodiment fusion
        vace_e_text_features: Optional[torch.Tensor] = None,
        vace_e_hand_motion_sequence: Optional[torch.Tensor] = None,
        vace_e_object_trajectory_sequence: Optional[torch.Tensor] = None,
        vace_e_object_ids: Optional[torch.Tensor] = None,
        vace_e_text_mask: Optional[torch.Tensor] = None,
        vace_e_motion_mask: Optional[torch.Tensor] = None,
        vace_e_trajectory_mask: Optional[torch.Tensor] = None,
        vace_e_embodiment_image_features: Optional[torch.Tensor] = None,
        vace_e_scale: Optional[float] = 1.0,
        # Randomness
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        # Shape
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames=81,
        # Classifier-free guidance
        cfg_scale: Optional[float] = 5.0,
        cfg_merge: Optional[bool] = False,
        # Scheduler
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,
        # Speed control
        motion_bucket_id: Optional[int] = None,
        # VAE tiling
        tiled: Optional[bool] = True,
        tile_size: Optional[tuple[int, int]] = (30, 52),
        tile_stride: Optional[tuple[int, int]] = (15, 26),
        # Sliding window
        sliding_window_size: Optional[int] = None,
        sliding_window_stride: Optional[int] = None,
        # Teacache
        tea_cache_l1_thresh: Optional[float] = None,
        tea_cache_model_id: Optional[str] = "",
        # progress_bar
        progress_bar_cmd=tqdm,
    ):
        """
        Generate video using the Wan Video pipeline.
        
        This is the main inference method supporting all generation modes:
        
        Generation Modes:
        1. Text-to-Video: Only provide 'prompt'
        2. Image-to-Video: Provide 'prompt' + 'input_image'  
        3. Video-to-Video: Provide 'prompt' + 'input_video'
        4. First-Last Frame: Provide 'prompt' + 'input_image' + 'end_image'
        5. Control Generation: Add 'control_video' for structure guidance
        6. VACE Editing: Use 'vace_*' parameters for video editing
        7. Camera Control: Use 'camera_control_*' for camera movement
        
        Args:
            prompt: Text description of desired video
            negative_prompt: Text describing what to avoid
            input_image: Starting image for I2V generation
            end_image: Ending image for FLF2V generation
            input_video: Input video for V2V transformation
            denoising_strength: How much to modify input (0=no change, 1=full generation)
            control_video: Structure/pose control video
            reference_image: Reference image for style/appearance
            camera_control_direction: Camera movement direction
            camera_control_speed: Speed of camera movement
            camera_control_origin: Initial camera parameters
            vace_video: Video for VACE editing
            vace_video_mask: Mask for VACE editing regions
            vace_reference_image: Reference for VACE editing
            vace_scale: Strength of VACE conditioning
            seed: Random seed for reproducibility
            rand_device: Device for random number generation
            height: Output video height
            width: Output video width  
            num_frames: Number of frames to generate
            cfg_scale: Classifier-free guidance strength
            cfg_merge: Merge positive/negative CFG in single batch
            num_inference_steps: Number of denoising steps
            sigma_shift: Noise schedule shift parameter
            motion_bucket_id: Motion intensity control
            tiled: Use tiled VAE processing for memory efficiency
            tile_size: Size of VAE tiles
            tile_stride: Stride between VAE tiles
            sliding_window_size: Size of temporal sliding window
            sliding_window_stride: Stride of temporal sliding window
            tea_cache_l1_thresh: TeaCache optimization threshold
            tea_cache_model_id: Model ID for TeaCache
            progress_bar_cmd: Progress bar display function
            
        Returns:
            List of PIL Images representing the generated video
        """
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)
        
        # Organize inputs for processing pipeline
        # Positive conditioning (for main generation)
        inputs_posi = {
            "prompt": prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
        }
        # Negative conditioning (for CFG)
        inputs_nega = {
            "negative_prompt": negative_prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
        }
        # Shared parameters (used for both positive and negative)
        inputs_shared = {
            "input_image": input_image,
            "end_image": end_image,
            "input_video": input_video, "denoising_strength": denoising_strength,
            "control_video": control_video, "reference_image": reference_image,
            "camera_control_direction": camera_control_direction, "camera_control_speed": camera_control_speed, "camera_control_origin": camera_control_origin,
            "vace_video": vace_video, "vace_video_mask": vace_video_mask, "vace_reference_image": vace_reference_image, "vace_scale": vace_scale,
            "vace_e_text_features": vace_e_text_features, "vace_e_hand_motion_sequence": vace_e_hand_motion_sequence, 
            "vace_e_object_trajectory_sequence": vace_e_object_trajectory_sequence, "vace_e_object_ids": vace_e_object_ids,
            "vace_e_text_mask": vace_e_text_mask, "vace_e_motion_mask": vace_e_motion_mask, "vace_e_trajectory_mask": vace_e_trajectory_mask,
            "vace_e_embodiment_image_features": vace_e_embodiment_image_features, "vace_e_scale": vace_e_scale,
            "seed": seed, "rand_device": rand_device,
            "height": height, "width": width, "num_frames": num_frames,
            "cfg_scale": cfg_scale, "cfg_merge": cfg_merge,
            "sigma_shift": sigma_shift,
            "motion_bucket_id": motion_bucket_id,
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
            "sliding_window_size": sliding_window_size, "sliding_window_stride": sliding_window_stride,
        }
        
        # Process inputs through modular pipeline units
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Main denoising loop
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)

            # Inference
            noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
            if cfg_scale != 1.0:
                if cfg_merge:
                    noise_pred_posi, noise_pred_nega = noise_pred_posi.chunk(2, dim=0)
                else:
                    noise_pred_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep)
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # Scheduler
            inputs_shared["latents"] = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"])
        
        # VACE (TODO: remove it)
        if vace_reference_image is not None:
            inputs_shared["latents"] = inputs_shared["latents"][:, :, 1:]

        # Decode
        self.load_models_to_device(['vae'])
        video = self.vae.decode(inputs_shared["latents"], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video = self.vae_output_to_video(video)
        self.load_models_to_device([])

        return video


# === PIPELINE UNIT SYSTEM ===
# The following classes implement a modular processing system where each unit
# handles a specific aspect of video generation (shape checking, encoding, etc.)

class PipelineUnit:
    """
    Base class for modular pipeline processing units.
    
    Each unit handles a specific aspect of the generation pipeline:
    - Input validation and preprocessing
    - Model-specific encoding
    - Feature extraction and conditioning
    - Advanced optimizations
    
    This modular design allows:
    - Easy addition of new features
    - Independent testing of components  
    - Flexible pipeline composition
    - Better code organization
    """
    def __init__(
        self,
        seperate_cfg: bool = False,
        take_over: bool = False,
        input_params: tuple[str] = None,
        input_params_posi: dict[str, str] = None,
        input_params_nega: dict[str, str] = None,
        onload_model_names: tuple[str] = None
    ):
        """
        Initialize pipeline unit.
        
        Args:
            seperate_cfg: Whether to process positive/negative prompts separately
            take_over: Whether this unit completely handles the processing
            input_params: Parameters to extract from shared inputs
            input_params_posi: Positive-specific parameter mapping
            input_params_nega: Negative-specific parameter mapping  
            onload_model_names: Models to load before processing
        """
        self.seperate_cfg = seperate_cfg
        self.take_over = take_over
        self.input_params = input_params
        self.input_params_posi = input_params_posi
        self.input_params_nega = input_params_nega
        self.onload_model_names = onload_model_names


    def process(self, pipe: WanVideoPipeline, inputs: dict, positive=True, **kwargs) -> dict:
        """
        Process inputs through this unit.
        
        Args:
            pipe: Pipeline instance
            inputs: Input dictionary
            positive: Whether processing positive conditioning
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of processed outputs
        """
        raise NotImplementedError("`process` is not implemented.")


class PipelineUnitRunner:
    """
    Orchestrates execution of pipeline units with proper input/output handling.
    
    Manages:
    - Parameter routing between units
    - Model loading for unit requirements
    - CFG-aware processing (positive/negative)
    - Input/output dictionary management
    """
    def __init__(self):
        pass

    def __call__(self, unit: PipelineUnit, pipe: WanVideoPipeline, inputs_shared: dict, inputs_posi: dict, inputs_nega: dict) -> tuple[dict, dict]:
        """
        Execute a pipeline unit with proper input routing.
        
        Args:
            unit: Unit to execute
            pipe: Pipeline instance
            inputs_shared: Shared parameters
            inputs_posi: Positive conditioning parameters
            inputs_nega: Negative conditioning parameters
            
        Returns:
            Updated (inputs_shared, inputs_posi, inputs_nega)
        """
        if unit.take_over:
            # Let the pipeline unit take over this function.
            inputs_shared, inputs_posi, inputs_nega = unit.process(pipe, inputs_shared=inputs_shared, inputs_posi=inputs_posi, inputs_nega=inputs_nega)
        elif unit.seperate_cfg:
            # Positive side
            processor_inputs = {name: inputs_posi.get(name_) for name, name_ in unit.input_params_posi.items()}
            if unit.input_params is not None:
                for name in unit.input_params:
                    processor_inputs[name] = inputs_shared.get(name)
            processor_outputs = unit.process(pipe, **processor_inputs)
            inputs_posi.update(processor_outputs)
            # Negative side
            if inputs_shared["cfg_scale"] != 1:
                processor_inputs = {name: inputs_nega.get(name_) for name, name_ in unit.input_params_nega.items()}
                if unit.input_params is not None:
                    for name in unit.input_params:
                        processor_inputs[name] = inputs_shared.get(name)
                processor_outputs = unit.process(pipe, **processor_inputs)
                inputs_nega.update(processor_outputs)
            else:
                inputs_nega.update(processor_outputs)
        else:
            processor_inputs = {name: inputs_shared.get(name) for name in unit.input_params}
            processor_outputs = unit.process(pipe, **processor_inputs)
            inputs_shared.update(processor_outputs)
        return inputs_shared, inputs_posi, inputs_nega


# === SPECIFIC PIPELINE UNITS ===

class WanVideoUnit_ShapeChecker(PipelineUnit):
    """
    Validates and adjusts input dimensions to meet model requirements.
    
    Ensures height, width, and frame count are compatible with:
    - VAE encoding factors (typically 8x downsampling)
    - Temporal compression (4x temporal downsampling)
    - Model architecture constraints
    """
    def __init__(self):
        super().__init__(input_params=("height", "width", "num_frames"))

    def process(self, pipe: WanVideoPipeline, height, width, num_frames):
        height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
        return {"height": height, "width": width, "num_frames": num_frames}


class WanVideoUnit_NoiseInitializer(PipelineUnit):
    """
    Initialize random noise tensor for diffusion process.
    
    Creates Gaussian noise in latent space with proper dimensions:
    - Spatial: (height//8, width//8) due to VAE encoding
    - Temporal: (num_frames-1)//4 + 1 due to temporal compression  
    - Channels: 16 latent channels for Wan models
    """
    def __init__(self):
        super().__init__(input_params=("height", "width", "num_frames", "seed", "rand_device", "vace_reference_image"))

    def process(self, pipe: WanVideoPipeline, height, width, num_frames, seed, rand_device, vace_reference_image):
        length = (num_frames - 1) // 4 + 1
        if vace_reference_image is not None:
            length += 1
        noise = pipe.generate_noise((1, 16, length, height//8, width//8), seed=seed, rand_device=rand_device)
        if vace_reference_image is not None:
            noise = torch.concat((noise[:, :, -1:], noise[:, :, :-1]), dim=2)
        return {"noise": noise}
    

class WanVideoUnit_InputVideoEmbedder(PipelineUnit):
    """
    Encode input videos into latent space using VAE.
    
    Handles:
    - Video-to-video generation (encode input video)
    - VACE reference frames (special handling)
    - Noise scheduling for denoising strength control
    - Training vs inference mode differences
    """
    def __init__(self):
        super().__init__(
            input_params=("input_video", "noise", "tiled", "tile_size", "tile_stride", "vace_reference_image"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_video, noise, tiled, tile_size, tile_stride, vace_reference_image):
        if input_video is None:
            return {"latents": noise}
        pipe.load_models_to_device(["vae"])
        input_video = pipe.preprocess_video(input_video)
        input_latents = pipe.vae.encode(input_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        if vace_reference_image is not None:
            vace_reference_image = pipe.preprocess_video([vace_reference_image])
            vace_reference_latents = pipe.vae.encode(vace_reference_image, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
            input_latents = torch.concat([vace_reference_latents, input_latents], dim=2)
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents}


class WanVideoUnit_PromptEmbedder(PipelineUnit):
    """
    Encode text prompts into embeddings using T5 text encoder.
    
    Handles:
    - Positive and negative prompt encoding separately
    - T5-XXL encoder for rich semantic understanding
    - Proper tokenization and padding
    - CFG-aware processing
    """
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt", "positive": "positive"},
            input_params_nega={"prompt": "negative_prompt", "positive": "positive"},
            onload_model_names=("text_encoder",)
        )

    def process(self, pipe: WanVideoPipeline, prompt, positive) -> dict:
        pipe.load_models_to_device(self.onload_model_names)
        prompt_emb = pipe.prompter.encode_prompt(prompt, positive=positive, device=pipe.device)
        return {"context": prompt_emb}


class WanVideoUnit_ImageEmbedder(PipelineUnit):
    """
    Encode input images for image-to-video generation.
    
    Handles:
    - CLIP image encoding for semantic conditioning
    - VAE image encoding for structural conditioning
    - First-last frame video generation (FLF2V)
    - Proper masking and positioning
    """
    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("image_encoder", "vae")
        )

    def process(self, pipe: WanVideoPipeline, input_image, end_image, num_frames, height, width, tiled, tile_size, tile_stride):
        if input_image is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        clip_context = pipe.image_encoder.encode_image([image])
        msk = torch.ones(1, num_frames, height//8, width//8, device=pipe.device)
        msk[:, 1:] = 0
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(pipe.device)
            vae_input = torch.concat([image.transpose(0,1), torch.zeros(3, num_frames-2, height, width).to(image.device), end_image.transpose(0,1)],dim=1)
            if pipe.dit.has_image_pos_emb:
                clip_context = torch.concat([clip_context, pipe.image_encoder.encode_image([end_image])], dim=1)
            msk[:, -1:] = 1
        else:
            vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)

        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]
        
        y = pipe.vae.encode([vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        clip_context = clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"clip_feature": clip_context, "y": y}

        
class WanVideoUnit_FunControl(PipelineUnit):
    """
    Handle control video conditioning (pose, depth, edges, etc.).
    
    Control videos provide structural guidance for generation:
    - Pose control: Human/animal pose sequences
    - Depth control: Depth map sequences  
    - Edge control: Edge/line art sequences
    - Custom control: User-defined control signals
    """
    def __init__(self):
        super().__init__(
            input_params=("control_video", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride", "clip_feature", "y"),
            onload_model_names=("vae")
        )

    def process(self, pipe: WanVideoPipeline, control_video, num_frames, height, width, tiled, tile_size, tile_stride, clip_feature, y):
        if control_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        control_video = pipe.preprocess_video(control_video)
        control_latents = pipe.vae.encode(control_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        control_latents = control_latents.to(dtype=pipe.torch_dtype, device=pipe.device)
        if clip_feature is None or y is None:
            clip_feature = torch.zeros((1, 257, 1280), dtype=pipe.torch_dtype, device=pipe.device)
            y = torch.zeros((1, 16, (num_frames - 1) // 4 + 1, height//8, width//8), dtype=pipe.torch_dtype, device=pipe.device)
        else:
            y = y[:, -16:]
        y = torch.concat([control_latents, y], dim=1)
        return {"clip_feature": clip_feature, "y": y}
    

class WanVideoUnit_FunReference(PipelineUnit):
    """
    Handle reference image conditioning for style/appearance control.
    
    Reference images provide:
    - Style guidance (artistic style, color palette)
    - Character consistency (face, clothing, appearance)
    - Scene consistency (lighting, atmosphere)
    - Object appearance (texture, materials)
    """
    def __init__(self):
        super().__init__(
            input_params=("reference_image", "height", "width", "reference_image"),
            onload_model_names=("vae")
        )

    def process(self, pipe: WanVideoPipeline, reference_image, height, width):
        if reference_image is None:
            return {}
        pipe.load_models_to_device(["vae"])
        reference_image = reference_image.resize((width, height))
        reference_latents = pipe.preprocess_video([reference_image])
        reference_latents = pipe.vae.encode(reference_latents, device=pipe.device)
        clip_feature = pipe.preprocess_image(reference_image)
        clip_feature = pipe.image_encoder.encode_image([clip_feature])
        return {"reference_latents": reference_latents, "clip_feature": clip_feature}


class WanVideoUnit_FunCameraControl(PipelineUnit):
    """
    Generate camera movement conditioning for cinematic effects.
    
    Camera controls include:
    - Pan movements (left, right, up, down)
    - Diagonal movements (combinations)
    - Zoom in/out effects
    - Speed control for movement
    - Custom camera trajectories
    """
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames", "camera_control_direction", "camera_control_speed", "camera_control_origin", "latents", "input_image")
        )

    def process(self, pipe: WanVideoPipeline, height, width, num_frames, camera_control_direction, camera_control_speed, camera_control_origin, latents, input_image):
        if camera_control_direction is None:
            return {}
        camera_control_plucker_embedding = pipe.dit.control_adapter.process_camera_coordinates(
            camera_control_direction, num_frames, height, width, camera_control_speed, camera_control_origin)
        
        control_camera_video = camera_control_plucker_embedding[:num_frames].permute([3, 0, 1, 2]).unsqueeze(0)
        control_camera_latents = torch.concat(
            [
                torch.repeat_interleave(control_camera_video[:, :, 0:1], repeats=4, dim=2),
                control_camera_video[:, :, 1:]
            ], dim=2
        ).transpose(1, 2)
        b, f, c, h, w = control_camera_latents.shape
        control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, 4, c, h, w).transpose(2, 3)
        control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, c * 4, h, w).transpose(1, 2)
        control_camera_latents_input = control_camera_latents.to(device=pipe.device, dtype=pipe.torch_dtype)

        input_image = input_image.resize((width, height))
        input_latents = pipe.preprocess_video([input_image])
        input_latents = pipe.vae.encode(input_latents, device=pipe.device)
        y = torch.zeros_like(latents).to(pipe.device)
        y[:, :, :1] = input_latents
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"control_camera_latents_input": control_camera_latents_input, "y": y}


class WanVideoUnit_SpeedControl(PipelineUnit):
    """
    Control motion speed/intensity in generated videos.
    
    Motion control parameters:
    - motion_bucket_id: Discrete speed levels (0=slow, higher=faster)
    - Continuous speed control through motion controller
    - Frame rate adjustment
    - Temporal consistency preservation
    """
    def __init__(self):
        super().__init__(input_params=("motion_bucket_id",))

    def process(self, pipe: WanVideoPipeline, motion_bucket_id):
        if motion_bucket_id is None:
            return {}
        motion_bucket_id = torch.Tensor((motion_bucket_id,)).to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"motion_bucket_id": motion_bucket_id}


class WanVideoUnit_VACE(PipelineUnit):
    """
    Video Editing and Completion Engine (VACE) processing.
    
    VACE capabilities:
    - Video inpainting: Fill missing regions using masks
    - Video outpainting: Extend video boundaries  
    - Object removal/addition: Edit specific objects
    - Style transfer: Change video style while preserving structure
    - Temporal consistency: Maintain coherence across frames
    """
    def __init__(self):
        super().__init__(
            input_params=("vace_video", "vace_video_mask", "vace_reference_image", "vace_scale", "height", "width", "num_frames", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(
        self,
        pipe: WanVideoPipeline,
        vace_video, vace_video_mask, vace_reference_image, vace_scale,
        height, width, num_frames,
        tiled, tile_size, tile_stride
    ):
        if vace_video is not None or vace_video_mask is not None or vace_reference_image is not None:
            pipe.load_models_to_device(["vae"])
            if vace_video is None:
                vace_video = torch.zeros((1, 3, num_frames, height, width), dtype=pipe.torch_dtype, device=pipe.device)
            else:
                vace_video = pipe.preprocess_video(vace_video)
            
            if vace_video_mask is None:
                vace_video_mask = torch.ones_like(vace_video)
            else:
                vace_video_mask = pipe.preprocess_video(vace_video_mask, min_value=0, max_value=1)
            
            inactive = vace_video * (1 - vace_video_mask) + 0 * vace_video_mask
            reactive = vace_video * vace_video_mask + 0 * (1 - vace_video_mask)
            inactive = pipe.vae.encode(inactive, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
            reactive = pipe.vae.encode(reactive, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
            vace_video_latents = torch.concat((inactive, reactive), dim=1)
            
            vace_mask_latents = rearrange(vace_video_mask[0,0], "T (H P) (W Q) -> 1 (P Q) T H W", P=8, Q=8)
            vace_mask_latents = torch.nn.functional.interpolate(vace_mask_latents, size=((vace_mask_latents.shape[2] + 3) // 4, vace_mask_latents.shape[3], vace_mask_latents.shape[4]), mode='nearest-exact')
            
            if vace_reference_image is None:
                pass
            else:
                vace_reference_image = pipe.preprocess_video([vace_reference_image])
                vace_reference_latents = pipe.vae.encode(vace_reference_image, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
                vace_reference_latents = torch.concat((vace_reference_latents, torch.zeros_like(vace_reference_latents)), dim=1)
                vace_video_latents = torch.concat((vace_reference_latents, vace_video_latents), dim=2)
                vace_mask_latents = torch.concat((torch.zeros_like(vace_mask_latents[:, :, :1]), vace_mask_latents), dim=2)
            
            vace_context = torch.concat((vace_video_latents, vace_mask_latents), dim=1)
            return {"vace_context": vace_context, "vace_scale": vace_scale}
        else:
            return {"vace_context": None, "vace_scale": vace_scale}


class WanVideoUnit_VACE_E(PipelineUnit):
    """
    Enhanced Video Editing and Completion Engine (VACE-E) processing.
    
    VACE-E capabilities:
    - Task-embodiment fusion for robot manipulation videos
    - Dual-hand motion sequence processing (left/right wrist poses + gripper states)
    - Object trajectory sequence processing with type embeddings
    - Text-motion-trajectory multi-modal fusion
    - CLIP-encoded end-effector image processing
    - Cross-attention alignment between task and embodiment features
    - Independent weighted fusion to reduce modality correlation
    
    Input Format Support:
    - 20D dual-hand: [left_wrist(9), right_wrist(9), left_gripper(1), right_gripper(1)]
    - 10D single-hand (legacy): [wrist(9), gripper(1)] - automatically converted to dual-hand
    """
    def __init__(self):
        super().__init__(
            input_params=(
                "vace_e_text_features", "vace_e_hand_motion_sequence", "vace_e_object_trajectory_sequence",
                "vace_e_object_ids", "vace_e_text_mask", "vace_e_motion_mask", "vace_e_trajectory_mask",
                "vace_e_embodiment_image_features", "vace_e_scale"
            ),
            onload_model_names=()  # VACE-E doesn't need VAE encoding
        )

    def _validate_and_convert_hand_motion(self, hand_motion_sequence):
        """
        Validate and convert hand motion sequence to dual-hand format.
        
        Args:
            hand_motion_sequence: Input tensor with shape [batch, seq_len, feature_dim]
                                 - 20D: [left_wrist(9), right_wrist(9), left_gripper(1), right_gripper(1)]
                                 - 10D: [wrist(9), gripper(1)] - legacy single-hand format
        
        Returns:
            torch.Tensor: Validated dual-hand motion sequence [batch, seq_len, 20]
        
        Raises:
            ValueError: If input format is invalid
        """
        if hand_motion_sequence is None:
            return None
            
        # Check input dimensions
        if len(hand_motion_sequence.shape) != 3:
            raise ValueError(f"Hand motion sequence must have 3 dimensions [batch, seq_len, feature_dim], got {hand_motion_sequence.shape}")
        
        batch_size, seq_len, feature_dim = hand_motion_sequence.shape
        
        if feature_dim == 20:
            # Already in dual-hand format: [left_wrist(9), right_wrist(9), left_gripper(1), right_gripper(1)]
            print(f"✅ Dual-hand motion sequence detected: {hand_motion_sequence.shape} (20D format)")
            return hand_motion_sequence
            
        elif feature_dim == 10:
            # Legacy single-hand format: [wrist(9), gripper(1)]
            # Convert to dual-hand: put single hand in right hand position, left hand as zeros
            print(f"⚠️ Legacy single-hand motion sequence detected: {hand_motion_sequence.shape} (10D format)")
            print(f"   Converting to dual-hand format (single hand → right hand, left hand → zeros)...")
            
            wrist_poses = hand_motion_sequence[:, :, :9]    # [batch, seq_len, 9]
            gripper_states = hand_motion_sequence[:, :, 9:] # [batch, seq_len, 1]
            
            # Create dual-hand with left hand as zeros, right hand with actual data
            device = hand_motion_sequence.device
            dtype = hand_motion_sequence.dtype
            left_wrist_zeros = torch.zeros_like(wrist_poses)     # Left wrist: all zeros
            left_gripper_zeros = torch.zeros_like(gripper_states) # Left gripper: closed (0)
            
            dual_hand_motion = torch.cat([
                left_wrist_zeros,   # Left wrist (9D): zeros
                wrist_poses,        # Right wrist (9D): actual single hand data
                left_gripper_zeros, # Left gripper (1D): closed (0)
                gripper_states      # Right gripper (1D): actual single hand data
            ], dim=-1)  # [batch, seq_len, 20]
            
            print(f"✅ Converted to dual-hand format: {dual_hand_motion.shape} (20D format)")
            print(f"   Left hand: zeros (inactive), Right hand: actual data (active)")
            return dual_hand_motion
            
        else:
            # Invalid format
            raise ValueError(
                f"Invalid hand motion sequence feature dimension: {feature_dim}. "
                f"Expected 20D (dual-hand: left_wrist(9) + right_wrist(9) + left_gripper(1) + right_gripper(1)) "
                f"or 10D (legacy single-hand: wrist(9) + gripper(1) for backward compatibility). "
                f"Input shape: {hand_motion_sequence.shape}"
            )

    def process(
        self,
        pipe: WanVideoPipeline,
        vace_e_text_features, vace_e_hand_motion_sequence, vace_e_object_trajectory_sequence,
        vace_e_object_ids, vace_e_text_mask, vace_e_motion_mask, vace_e_trajectory_mask,
        vace_e_embodiment_image_features, vace_e_scale
    ):
        # Check if any VACE-E features are provided
        has_task_features = (vace_e_text_features is not None or 
                           vace_e_hand_motion_sequence is not None or 
                           vace_e_object_trajectory_sequence is not None)
        has_embodiment_features = vace_e_embodiment_image_features is not None
        
        if has_task_features or has_embodiment_features:
            # Validate and convert hand motion sequence to dual-hand format
            validated_hand_motion = self._validate_and_convert_hand_motion(vace_e_hand_motion_sequence)
            
            # Prepare VACE-E context with validated features
            # The actual processing will be done in the VACE-E model during forward pass
            vace_e_context = {
                "text_features": vace_e_text_features,
                "hand_motion_sequence": validated_hand_motion,  # Use validated/converted version
                "object_trajectory_sequence": vace_e_object_trajectory_sequence,
                "object_ids": vace_e_object_ids,
                "text_mask": vace_e_text_mask,
                "motion_mask": vace_e_motion_mask,
                "trajectory_mask": vace_e_trajectory_mask,
                "embodiment_image_features": vace_e_embodiment_image_features,
            }
            return {"vace_e_context": vace_e_context, "vace_e_scale": vace_e_scale}
        else:
            return {"vace_e_context": None, "vace_e_scale": vace_e_scale}


class WanVideoUnit_UnifiedSequenceParallel(PipelineUnit):
    """
    Configure Unified Sequence Parallel processing.
    
    USP enables efficient distributed processing by:
    - Splitting long sequences across multiple GPUs
    - Ring-based communication for memory efficiency
    - Overlap computation and communication
    - Scale to very long videos (hundreds of frames)
    """
    def __init__(self):
        super().__init__(input_params=())

    def process(self, pipe: WanVideoPipeline):
        if hasattr(pipe, "use_unified_sequence_parallel"):
            if pipe.use_unified_sequence_parallel:
                return {"use_unified_sequence_parallel": True}
        return {}


class WanVideoUnit_TeaCache(PipelineUnit):
    """
    Configure TeaCache optimization for faster inference.
    
    TeaCache reduces computation by:
    - Caching attention weights across timesteps
    - Reusing similar computations when inputs are similar
    - Adaptive caching based on input change thresholds
    - Model-specific optimization coefficients
    """
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"num_inference_steps": "num_inference_steps", "tea_cache_l1_thresh": "tea_cache_l1_thresh", "tea_cache_model_id": "tea_cache_model_id"},
            input_params_nega={"num_inference_steps": "num_inference_steps", "tea_cache_l1_thresh": "tea_cache_l1_thresh", "tea_cache_model_id": "tea_cache_model_id"},
        )

    def process(self, pipe: WanVideoPipeline, num_inference_steps, tea_cache_l1_thresh, tea_cache_model_id):
        if tea_cache_l1_thresh is None:
            return {}
        return {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id)}


class WanVideoUnit_CfgMerger(PipelineUnit):
    """
    Merge positive and negative conditioning for efficient CFG processing.
    
    When cfg_merge=True:
    - Concatenates positive and negative inputs into single batch
    - Processes both conditioning types in one forward pass
    - Reduces VRAM usage and computation time
    - Splits outputs back for CFG calculation
    """
    def __init__(self):
        super().__init__(take_over=True)
        self.concat_tensor_names = ["context", "clip_feature", "y", "reference_latents"]

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        if not inputs_shared["cfg_merge"]:
            return inputs_shared, inputs_posi, inputs_nega
        for name in self.concat_tensor_names:
            tensor_posi = inputs_posi.get(name)
            tensor_nega = inputs_nega.get(name)
            tensor_shared = inputs_shared.get(name)
            if tensor_posi is not None and tensor_nega is not None:
                inputs_shared[name] = torch.concat((tensor_posi, tensor_nega), dim=0)
            elif tensor_shared is not None:
                inputs_shared[name] = torch.concat((tensor_shared, tensor_shared), dim=0)
        inputs_posi.clear()
        inputs_nega.clear()
        return inputs_shared, inputs_posi, inputs_nega


# === OPTIMIZATION CLASSES ===

class TeaCache:
    """
    TeaCache: Temporal Attention Caching for faster video generation.
    
    This optimization technique caches attention computations when:
    1. Input changes are below a threshold (indicating similarity)
    2. Previous computations can be reused
    3. Model-specific coefficients determine caching effectiveness
    
    Benefits:
    - 2-3x speedup for video generation
    - Maintains quality while reducing computation
    - Adaptive thresholding based on content changes
    - Model-specific optimization parameters
    """
    def __init__(self, num_inference_steps, rel_l1_thresh, model_id):
        """
        Initialize TeaCache with model-specific parameters.
        
        Args:
            num_inference_steps: Total denoising steps
            rel_l1_thresh: Threshold for caching decision
            model_id: Model ID for coefficient lookup
        """
        self.num_inference_steps = num_inference_steps
        self.step = 0
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.rel_l1_thresh = rel_l1_thresh
        self.previous_residual = None
        self.previous_hidden_states = None
        
        # Model-specific optimization coefficients
        self.coefficients_dict = {
            "Wan2.1-T2V-1.3B": [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02],
            "Wan2.1-T2V-14B": [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01],
            "Wan2.1-I2V-14B-480P": [2.57151496e+05, -3.54229917e+04,  1.40286849e+03, -1.35890334e+01, 1.32517977e-01],
            "Wan2.1-I2V-14B-720P": [ 8.10705460e+03,  2.13393892e+03, -3.72934672e+02,  1.66203073e+01, -4.17769401e-02],
        }
        if model_id not in self.coefficients_dict:
            supported_model_ids = ", ".join([i for i in self.coefficients_dict])
            raise ValueError(f"{model_id} is not a supported TeaCache model id. Please choose a valid model id in ({supported_model_ids}).")
        self.coefficients = self.coefficients_dict[model_id]

    def check(self, dit: WanModel, x, t_mod):
        """
        Check whether to use cached computation or compute fresh.
        
        Args:
            dit: DiT model instance
            x: Current hidden states
            t_mod: Time modulation tensor
            
        Returns:
            True if should use cache, False if should compute
        """
        modulated_inp = t_mod.clone()
        if self.step == 0 or self.step == self.num_inference_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = self.coefficients
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.step += 1
        if self.step == self.num_inference_steps:
            self.step = 0
        if should_calc:
            self.previous_hidden_states = x.clone()
        return not should_calc

    def store(self, hidden_states):
        """Store residual for cache updates."""
        self.previous_residual = hidden_states - self.previous_hidden_states
        self.previous_hidden_states = None

    def update(self, hidden_states):
        """Update hidden states using cached residual."""
        hidden_states = hidden_states + self.previous_residual
        return hidden_states


class TemporalTiler_BCTHW:
    """
    Temporal tiling for processing long videos with sliding windows.
    
    This technique processes long videos in overlapping segments:
    1. Split video into overlapping temporal windows
    2. Process each window independently
    3. Blend overlapping regions with proper weighting
    4. Reconstruct full video from processed segments
    
    Benefits:
    - Process arbitrarily long videos
    - Maintain temporal consistency
    - Reduce peak memory usage
    - Parallel processing of segments
    """
    def __init__(self):
        pass

    def build_1d_mask(self, length, left_bound, right_bound, border_width):
        """Build 1D blending mask for segment boundaries."""
        x = torch.ones((length,))
        if not left_bound:
            x[:border_width] = (torch.arange(border_width) + 1) / border_width
        if not right_bound:
            x[-border_width:] = torch.flip((torch.arange(border_width) + 1) / border_width, dims=(0,))
        return x

    def build_mask(self, data, is_bound, border_width):
        """Build temporal blending mask for current segment."""
        _, _, T, _, _ = data.shape
        t = self.build_1d_mask(T, is_bound[0], is_bound[1], border_width[0])
        mask = repeat(t, "T -> 1 1 T 1 1")
        return mask
    
    def run(self, model_fn, sliding_window_size, sliding_window_stride, computation_device, computation_dtype, model_kwargs, tensor_names, batch_size=None):
        """
        Process video using sliding window with blending.
        
        Args:
            model_fn: Model function to apply
            sliding_window_size: Size of temporal window
            sliding_window_stride: Stride between windows
            computation_device: Device for computation
            computation_dtype: Data type for computation
            model_kwargs: Model arguments
            tensor_names: Names of tensors to tile
            batch_size: Batch size multiplier for CFG
            
        Returns:
            Processed video tensor
        """
        tensor_names = [tensor_name for tensor_name in tensor_names if model_kwargs.get(tensor_name) is not None]
        tensor_dict = {tensor_name: model_kwargs[tensor_name] for tensor_name in tensor_names}
        B, C, T, H, W = tensor_dict[tensor_names[0]].shape
        if batch_size is not None:
            B *= batch_size
        data_device, data_dtype = tensor_dict[tensor_names[0]].device, tensor_dict[tensor_names[0]].dtype
        value = torch.zeros((B, C, T, H, W), device=data_device, dtype=data_dtype)
        weight = torch.zeros((1, 1, T, 1, 1), device=data_device, dtype=data_dtype)
        for t in range(0, T, sliding_window_stride):
            if t - sliding_window_stride >= 0 and t - sliding_window_stride + sliding_window_size >= T:
                continue
            t_ = min(t + sliding_window_size, T)
            model_kwargs.update({
                tensor_name: tensor_dict[tensor_name][:, :, t: t_:, :].to(device=computation_device, dtype=computation_dtype) \
                    for tensor_name in tensor_names
            })
            model_output = model_fn(**model_kwargs).to(device=data_device, dtype=data_dtype)
            mask = self.build_mask(
                model_output,
                is_bound=(t == 0, t_ == T),
                border_width=(sliding_window_size - sliding_window_stride,)
            ).to(device=data_device, dtype=data_dtype)
            value[:, :, t: t_, :, :] += model_output * mask
            weight[:, :, t: t_, :, :] += mask
        value /= weight
        model_kwargs.update(tensor_dict)
        return value


def model_fn_wan_video(
    dit: WanModel,
    motion_controller: WanMotionControllerModel = None,
    vace: VaceWanModel = None,
    vace_e: VaceWanModelE = None,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    reference_latents = None,
    vace_context = None,
    vace_scale = 1.0,
    vace_e_context = None,
    vace_e_scale = 1.0,
    tea_cache: TeaCache = None,
    use_unified_sequence_parallel: bool = False,
    motion_bucket_id: Optional[torch.Tensor] = None,
    sliding_window_size: Optional[int] = None,
    sliding_window_stride: Optional[int] = None,
    cfg_merge: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    control_camera_latents_input = None,
    **kwargs,
):
    """
    Core model function for Wan Video generation.
    
    This function orchestrates the diffusion transformer (DiT) and associated
    components to perform one denoising step. It handles:
    
    1. Input preprocessing and conditioning
    2. Temporal tiling for long videos
    3. Model forward pass with various conditionings
    4. Advanced optimizations (TeaCache, USP, gradient checkpointing)
    5. Output postprocessing
    
    Args:
        dit: Diffusion Transformer model
        motion_controller: Motion dynamics controller
        vace: Video editing model
        vace_e: Enhanced video editing model
        latents: Noisy latent tensor to denoise
        timestep: Current diffusion timestep
        context: Text conditioning from T5 encoder
        clip_feature: Image conditioning from CLIP encoder
        y: Additional conditioning (masks, control signals)
        reference_latents: Reference image latents
        vace_context: VACE editing context
        vace_scale: VACE conditioning strength
        vace_e_context: VACE-E editing context
        vace_e_scale: VACE-E conditioning strength
        tea_cache: TeaCache optimization instance
        use_unified_sequence_parallel: Enable USP
        motion_bucket_id: Motion speed control
        sliding_window_size: Temporal window size
        sliding_window_stride: Temporal window stride
        cfg_merge: Whether positive/negative are merged
        use_gradient_checkpointing: Enable gradient checkpointing
        use_gradient_checkpointing_offload: Offload checkpointed activations
        control_camera_latents_input: Camera control conditioning
        **kwargs: Additional arguments
        
    Returns:
        Predicted noise/flow direction tensor
    """
    # Use sliding window for long videos
    if sliding_window_size is not None and sliding_window_stride is not None:
        model_kwargs = dict(
            dit=dit,
            motion_controller=motion_controller,
            vace=vace,
            vace_e=vace_e,
            latents=latents,
            timestep=timestep,
            context=context,
            clip_feature=clip_feature,
            y=y,
            reference_latents=reference_latents,
            vace_context=vace_context,
            vace_scale=vace_scale,
            vace_e_context=vace_e_context,
            vace_e_scale=vace_e_scale,
            tea_cache=tea_cache,
            use_unified_sequence_parallel=use_unified_sequence_parallel,
            motion_bucket_id=motion_bucket_id,
        )
        return TemporalTiler_BCTHW().run(
            model_fn_wan_video,
            sliding_window_size, sliding_window_stride,
            latents.device, latents.dtype,
            model_kwargs=model_kwargs,
            tensor_names=["latents", "y"],
            batch_size=2 if cfg_merge else 1
        )
    
    # Initialize USP if enabled
    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (get_sequence_parallel_rank,
                                            get_sequence_parallel_world_size,
                                            get_sp_group)
    
    # Process time conditioning
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    
    # Add motion control if available
    if motion_bucket_id is not None and motion_controller is not None:
        t_mod = t_mod + motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim))
    
    # Process text conditioning
    context = dit.text_embedding(context)

    x = latents
    # Handle merged CFG (positive and negative in same batch)
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)
    
    # Add image conditioning if available
    if dit.has_image_input:
        x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)
    
    # Patchify inputs and add camera control
    x, (f, h, w) = dit.patchify(x, control_camera_latents_input)

    # Add reference image conditioning
    if reference_latents is not None:
        if len(reference_latents.shape) == 5:
            reference_latents = reference_latents[:, :, 0]
        reference_latents = dit.ref_conv(reference_latents).flatten(2).transpose(1, 2)
        x = torch.concat([reference_latents, x], dim=1)
        f += 1
    
    # Generate positional embeddings
    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
    
    # TeaCache optimization check
    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, x, t_mod)
    else:
        tea_cache_update = False
        
    # VACE conditioning preparation
    if vace_context is not None:
        vace_hints = vace(x, vace_context, context, t_mod, freqs)
    
    # VACE-E conditioning preparation (enhanced with task-embodiment fusion)
    if vace_e_context is not None and vace_e is not None:
        # Ensure all tensors in VACE-E context are on the correct device
        device = next(vace_e.parameters()).device
        vace_e_context_device = {}
        for key, value in vace_e_context.items():
            if isinstance(value, torch.Tensor):
                vace_e_context_device[key] = value.to(device)
            else:
                vace_e_context_device[key] = value
        
        # Check if we need to return intermediate features for CLUB loss
        return_intermediate_features = vace_e_context_device.get("return_intermediate_features", False)
        
        # Extract task and embodiment features from VACE-E context
        vace_e_result = vace_e(
            x, None, context, t_mod, freqs,  # x, vace_context (None for VACE-E), context, t_mod, freqs
            # Task features
            text_features=vace_e_context_device.get("text_features"),
            hand_motion_sequence=vace_e_context_device.get("hand_motion_sequence"),
            object_trajectory_sequence=vace_e_context_device.get("object_trajectory_sequence"),
            object_ids=vace_e_context_device.get("object_ids"),
            text_mask=vace_e_context_device.get("text_mask"),
            motion_mask=vace_e_context_device.get("motion_mask"),
            trajectory_mask=vace_e_context_device.get("trajectory_mask"),
            # Embodiment features
            embodiment_image_features=vace_e_context_device.get("embodiment_image_features"),
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            # CLUB training support
            return_intermediate_features=return_intermediate_features,
        )
        
        # Handle different return formats
        if return_intermediate_features:
            vace_e_hints, task_features, embodiment_features = vace_e_result
            # Store features for CLUB loss computation
            # We store these as global variables that can be accessed by the training_loss function
            globals()['_current_task_features'] = task_features
            globals()['_current_embodiment_features'] = embodiment_features
        else:
            vace_e_hints = vace_e_result
    
    # Process through transformer blocks
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
    if tea_cache_update:
        # Use cached computation
        x = tea_cache.update(x)
    else:
        # Full computation through transformer blocks
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        
        for block_id, block in enumerate(dit.blocks):
            if use_gradient_checkpointing_offload:
                # Gradient checkpointing with CPU offload
                with torch.autograd.graph.save_on_cpu():
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs,
                        use_reentrant=False,
                    )
            elif use_gradient_checkpointing:
                # Standard gradient checkpointing
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, context, t_mod, freqs,
                    use_reentrant=False,
                )
            else:
                # Normal forward pass
                x = block(x, context, t_mod, freqs)
                
            # Apply VACE hints at specific layers
            if vace_context is not None and block_id in vace.vace_layers_mapping:
                current_vace_hint = vace_hints[vace.vace_layers_mapping[block_id]]
                if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
                    current_vace_hint = torch.chunk(current_vace_hint, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
                x = x + current_vace_hint * vace_scale
                
            # Apply VACE-E hints at specific layers (enhanced with task-embodiment fusion)
            if vace_e_context is not None and vace_e is not None and block_id in vace_e.vace_layers_mapping:
                current_vace_e_hint = vace_e_hints[vace_e.vace_layers_mapping[block_id]]
                if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
                    current_vace_e_hint = torch.chunk(current_vace_e_hint, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
                x = x + current_vace_e_hint * vace_e_scale
        if tea_cache is not None:
            tea_cache.store(x)
            
    # Final output projection
    x = dit.head(x, t)
    
    # Gather USP results
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = get_sp_group().all_gather(x, dim=1)
            
    # Remove reference latents
    if reference_latents is not None:
        x = x[:, reference_latents.shape[1]:]
        f -= 1
    x = dit.unpatchify(x, (f, h, w))
    return x


# === VACE-E Usage Examples ===

def example_vace_e_usage():
    """
    Example demonstrating how to use VACE-E (Enhanced Video Animation and Control Engine)
    with the WanVideoPipeline for robot manipulation video generation.
    
    This example shows the complete workflow:
    1. Load the enhanced WanVideoPipeline with VACE-E support
    2. Prepare task features (text, hand motion, object trajectory)
    3. Prepare embodiment features (CLIP-encoded end-effector image)
    4. Generate robot manipulation video with task-embodiment guidance
    """
    import torch
    from PIL import Image
    
    # Example: Load pipeline with VACE-E support
    print("=== VACE-E Enhanced Video Generation Example ===")
    print("\n1. Loading WanVideoPipeline with VACE-E support:")
    print("""
    from diffsynth.pipelines.wan_video_new_E import WanVideoPipeline
    from diffsynth.models.wan_video_vace_E import create_vace_model_from_dit
    from diffsynth.configs.model_config import ModelConfig
    
    # Configure model loading
    model_configs = [
        ModelConfig(model_id='Wan-AI/Wan2.1-T2V-1.3B', origin_file_pattern='*dit*'),
        ModelConfig(model_id='Wan-AI/Wan2.1-T2V-1.3B', origin_file_pattern='*vae*'),
        ModelConfig(model_id='Wan-AI/Wan2.1-T2V-1.3B', origin_file_pattern='*text*'),
    ]
    
    # Load enhanced pipeline
    pipe = WanVideoPipeline.from_pretrained(
        model_configs=model_configs,
        device='cuda',
        torch_dtype=torch.bfloat16,
        # VACE-E configuration (automatically initialized with DiT weights)
        enable_vace_e=True,
        vace_e_layers=(0, 5, 10, 15, 20, 25),  # Which DiT layers receive VACE-E hints
        vace_e_task_processing=True  # Enable task-embodiment fusion
    )
    
    # VACE-E model is automatically created and initialized!
    # No need for manual initialization anymore
    """)
    
    print("\n   Alternative configurations:")
    print("""
    # Disable VACE-E for standard video generation
    pipe = WanVideoPipeline.from_pretrained(
        model_configs=model_configs,
        enable_vace_e=False
    )
    
    # Custom VACE-E layer configuration for fine-grained control
    pipe = WanVideoPipeline.from_pretrained(
        model_configs=model_configs,
        enable_vace_e=True,
        vace_e_layers=(0, 3, 6, 9, 12),  # Custom layer selection
        vace_e_task_processing=True
    )
    
    # For models without task processing (embodiment only)
    pipe = WanVideoPipeline.from_pretrained(
        model_configs=model_configs,
        enable_vace_e=True,
        vace_e_task_processing=False  # Only embodiment features
    )
    """)
    
    print("\n2. Preparing task features (text, motion, trajectory):")
    print("""
    # Task 1: Text description
    prompt = "Pick up the red cube and place it in the blue bowl"
    text_features = pipe.prompter.encode_prompt(prompt, device=pipe.device)
    
    # Task 2: Dual-hand motion sequence (NEW: 20D format for realistic robot manipulation)
    # Format: [batch_size, sequence_length, 20] where 20 = dual-hand format:
    #   - left_wrist(9):  Left hand 3D position + 6D rotation
    #   - right_wrist(9): Right hand 3D position + 6D rotation  
    #   - left_gripper(1): Left gripper state (0=closed, 1=open)
    #   - right_gripper(1): Right gripper state (0=closed, 1=open)
    
    sequence_length = 100
    
    # Method 1: Create dual-hand motion from separate left/right data
    left_wrist_poses = torch.randn(1, sequence_length, 9, device=pipe.device)      # Left hand poses
    right_wrist_poses = torch.randn(1, sequence_length, 9, device=pipe.device)     # Right hand poses
    left_gripper_states = torch.randint(0, 2, (1, sequence_length, 1), device=pipe.device).float()   # Left gripper
    right_gripper_states = torch.randint(0, 2, (1, sequence_length, 1), device=pipe.device).float()  # Right gripper
    
    # Combine into 20D dual-hand format
    hand_motion_sequence = torch.cat([
        left_wrist_poses,     # First 9 dims: left wrist pose
        right_wrist_poses,    # Next 9 dims: right wrist pose
        left_gripper_states,  # 19th dim: left gripper state
        right_gripper_states  # 20th dim: right gripper state
    ], dim=-1)  # Shape: [1, sequence_length, 20]
    
    # Method 2: Load from robot demonstration data (recommended)
    # hand_motion_sequence = load_dual_hand_data_from_hdf5("episode.hdf5")  # Pre-formatted 20D
    
    # Backward Compatibility: 10D single-hand format still supported
    # If you have legacy 10D data [wrist(9) + gripper(1)], it will be automatically
    # converted to 20D by placing the single hand in the right hand position and 
    # setting the left hand to zeros (inactive state).
    # single_hand_motion = torch.randn(1, sequence_length, 10)  # Legacy format
    # This gets converted to: [left_zeros(9), right_data(9), left_closed(1), right_data(1)]
    
    # Task 3: Object trajectory sequence (multiple objects)
    num_objects = 2  # Red cube + blue bowl
    object_trajectory_sequence = torch.randn(1, sequence_length, num_objects, 9, device=pipe.device)
    object_ids = torch.tensor([[0, 1]], device=pipe.device)  # Object type IDs
    
    # Create attention masks
    text_mask = torch.ones(1, text_features.shape[1], device=pipe.device).bool()
    motion_mask = torch.ones(1, sequence_length, device=pipe.device).bool()
    trajectory_mask = torch.ones(1, sequence_length, num_objects, device=pipe.device).bool()
    """)
    
    print("\n3. Preparing embodiment features (end-effector image):")
    print("""
    # Load end-effector image showing robot gripper
    end_effector_image = Image.open('path/to/robot_gripper.jpg')
    
    # Encode with CLIP (this provides embodiment context)
    clip_features = pipe.image_encoder.encode_image([end_effector_image])
    # Output: [1, 257, 1280] - CLIP standard format
    """)
    
    print("\n4. Generate robot manipulation video:")
    print("""
    # Enhanced video generation with task-embodiment fusion
    video = pipe(
        # Basic generation parameters
        prompt=prompt,
        height=480,
        width=832,
        num_frames=81,
        seed=42,
        
        # VACE-E task features
        vace_e_text_features=text_features,
        vace_e_hand_motion_sequence=hand_motion_sequence,
        vace_e_object_trajectory_sequence=object_trajectory_sequence,
        vace_e_object_ids=object_ids,
        vace_e_text_mask=text_mask,
        vace_e_motion_mask=motion_mask,
        vace_e_trajectory_mask=trajectory_mask,
        
        # VACE-E embodiment features
        vace_e_embodiment_image_features=clip_features,
        vace_e_scale=1.0,  # Conditioning strength
        
        # Standard generation parameters
        cfg_scale=5.0,
        num_inference_steps=50,
    )
    
    # Save generated robot manipulation video
    from diffsynth import save_video
    save_video(video, "robot_manipulation.mp4", fps=15, quality=5)
    """)
    
    print("\n✅ VACE-E Integration Benefits:")
    print("• Task-aware video generation: Considers text, motion, and trajectory context")
    print("• Dual-hand embodiment: Realistic robotic manipulation with coordinated left/right hands")
    print("• Embodiment-aware generation: Uses end-effector image for robot-specific content")
    print("• Independent fusion: Reduces correlation between task and embodiment modalities")
    print("• Seamless integration: Works with existing WanVideoPipeline infrastructure")
    print("• Scalable architecture: Supports different model sizes (1.3B, 14B, etc.)")
    
    print("\n🔧 Key Implementation Features:")
    print("• Dual-hand motion processing: 20D format for realistic robot coordination")
    print("• Backward compatibility: Automatic conversion from legacy 10D single-hand format")
    print("• Parallel processing: VACE and VACE-E work together, not replacing each other")
    print("• VRAM efficient: Automatic model management and gradient checkpointing support")
    print("• Flexible inputs: All task and embodiment features are optional")
    print("• DiT weight initialization: VACE-E blocks start with pre-trained DiT weights")
    
    return True


def example_club_loss_training():
    """
    Example of training with CLUB loss for mutual information minimization.
    
    This example demonstrates how to train the VACE-E model with disentangled
    representation learning using CLUB (Contrastive Log-ratio Upper Bound) to
    minimize mutual information between task and embodiment features.
    
    CLUB Training Process (following the official implementation):
    1. Phase 1: Train CLUB estimator using learning_loss() to approximate q_θ(embodiment|task)
    2. Phase 2: Use trained estimator with forward() to compute MI upper bound
    3. Minimize the MI upper bound in the main training loss
    
    Based on the vCLUB algorithm for learning disentangled representations.
    """
    import torch
    import torch.optim as optim
    
    print("=== CLUB Loss Training Example ===")
    print("\nTraining VACE-E with disentangled task-embodiment representations:")
    
    # Initialize pipeline with VACE-E
    pipe = WanVideoPipeline.from_pretrained(
        device="cuda",
        torch_dtype=torch.bfloat16,
        enable_vace_e=True,
        vace_e_layers=(0, 5, 10, 15, 20, 25),
        vace_e_task_processing=True,
    )
    
    # Configure CLUB loss parameters
    print("\n1. Configuring CLUB loss parameters:")
    pipe.configure_club_loss(
        lambda_weight=1.0,  # Weight for CLUB loss in total loss
        update_freq=1,      # Update CLUB estimator every step
        training_steps=5,   # Number of CLUB training steps per update
        club_lr=1e-3,       # Learning rate for CLUB optimizer
        enable=True         # Enable CLUB loss computation
    )
    
    # Create training data with task and embodiment features
    print("\n2. Preparing training data with VACE-E features:")
    batch_size = 2
    seq_len = 81
    
    training_inputs = {
        "input_latents": torch.randn(batch_size, 16, 21, 30, 52, device="cuda"),
        "noise": torch.randn(batch_size, 16, 21, 30, 52, device="cuda"),
        "context": torch.randn(batch_size, 77, 4096, device="cuda"),
        "vace_e_context": {
            # Task features (what to do)
            "text_features": torch.randn(batch_size, 77, 4096, device="cuda"),
            "hand_motion_sequence": torch.randn(batch_size, seq_len, 20, device="cuda"),
            "object_trajectory_sequence": torch.randn(batch_size, seq_len, 2, 9, device="cuda"),
            "object_ids": torch.tensor([[1, 2], [1, 2]], device="cuda"),
            "text_mask": torch.ones(batch_size, 77, device="cuda"),
            "motion_mask": torch.ones(batch_size, seq_len, device="cuda"),
            "trajectory_mask": torch.ones(batch_size, seq_len, 2, device="cuda"),
            # Embodiment features (how to do it)
            "embodiment_image_features": torch.randn(batch_size, 257, 1280, device="cuda"),
        }
    }
    
    # Setup optimizers
    print("\n3. Setting up optimizers:")
    main_optimizer = optim.AdamW(pipe.dit.parameters(), lr=1e-4)
    # CLUB optimizer will be automatically initialized when CLUB estimator is created
    
    # Training loop
    print("\n4. Training with CLUB loss:")
    print("Format: Step | Total Loss | Flow Loss | CLUB Loss | MI Upper Bound")
    print("-" * 70)
    
    for step in range(10):  # Demo with 10 steps
        main_optimizer.zero_grad()
        
        # Compute training loss with CLUB (get detailed breakdown for monitoring)
        loss_dict = pipe.training_loss(training_step=step, return_detailed_losses=True, **training_inputs)
        
        total_loss = loss_dict['total_loss']
        flow_loss = loss_dict['flow_loss']
        club_loss = loss_dict['club_loss']
        
        # Check if CLUB estimator was initialized
        if pipe.club_estimator is not None and step == 0:
            print(f"   🎯 CLUB estimator and optimizer automatically initialized")
        
        # Backward pass and optimization (CLUB training is handled automatically)
        total_loss.backward()
        main_optimizer.step()
        
        # Print progress
        mi_bound = club_loss.item() / pipe.club_lambda if pipe.club_lambda > 0 else 0
        print(f"   {step:2d}   | {total_loss:.4f}   | {flow_loss:.4f}   | {club_loss:.4f}   | {mi_bound:.4f}")
    
    # Display CLUB estimator information
    print("\n5. CLUB Estimator Final Information:")
    club_info = pipe.get_club_estimator_info()
    for key, value in club_info.items():
        print(f"   {key}: {value}")
    
    print("\n6. Key Benefits of CLUB Loss Training:")
    print("   • Two-Phase Training: First train q_θ(embodiment|task), then minimize MI upper bound")
    print("   • Disentangled Representations: Task and embodiment features become less correlated")
    print("   • Better Generalization: Model learns to separate 'what to do' from 'how to do it'")
    print("   • Controllable Generation: Independent control over task and embodiment aspects")
    print("   • Principled Approach: Based on mutual information theory with theoretical guarantees")
    print("   • Automatic Training: CLUB estimator is trained automatically during main training")
    
    print("\n✅ CLUB loss training example completed successfully!")
    return pipe


if __name__ == "__main__":
    # Run examples when script is executed directly
    print("Running VACE-E usage example...")
    example_vace_e_usage()
    
    print("\n" + "="*80 + "\n")
    
    print("Running CLUB loss training example...")
    example_club_loss_training()
