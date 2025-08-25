import torch, os, json
from diffsynth import load_state_dict
from diffsynth.pipelines.wan_video_new_E import WanVideoPipeline, ModelConfig
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, launch_training_task, wan_parser, enable_club_training_defaults
from dataset_E import VideoDatasetE, create_training_dataset
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
import warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class WanTrainingModuleE(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        # VACE-E specific parameters
        enable_vace_e=True,
        vace_e_layers=(0, 5, 10, 15, 20, 25),
        vace_e_task_processing=True,
        resume_path=None,
    ):
        super().__init__()
        # Load models (following exact train.py pattern)
        model_configs = []
        if model_paths is not None:
            model_paths = json.loads(model_paths)
            model_configs += [ModelConfig(path=path) for path in model_paths]
        if model_id_with_origin_paths is not None:
            model_id_with_origin_paths = model_id_with_origin_paths.split(",")
            model_configs += [ModelConfig(model_id=i.split(":")[0], origin_file_pattern=i.split(":")[1]) for i in model_id_with_origin_paths]
        
        # # Add CLIP model configuration for embodiment image encoding
        # model_configs.append(ModelConfig(
        #     model_id="Wan-AI/Wan2.1-I2V-14B-480P", 
        #     origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", 
        #     offload_device="cpu"
        # ))
        
        # Get device from accelerator if available, otherwise use cuda
        device = getattr(self, '_device', 'cuda')
        
        # Initialize VACE-E enhanced pipeline
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16, 
            device=device, 
            model_configs=model_configs,
            # VACE-E configuration
            enable_vace_e=enable_vace_e,
            vace_e_layers=vace_e_layers,
            vace_e_task_processing=vace_e_task_processing,
        )

        if resume_path:
            state_dict = load_state_dict(resume_path)
            self.pipe.vace_e.load_state_dict(state_dict)
        
        # Reset training scheduler
        self.pipe.scheduler.set_timesteps(1000, training=True)
        
        # Freeze untrainable models (include vace_e in potential trainable models)
        # Default to training dit and vace_e for VACE-E training
        if trainable_models is None:
            if enable_vace_e:
                trainable_models = "dit,vace_e"  # Train both DiT and VACE-E for task-embodiment fusion
            else:
                trainable_models = "dit"  # Default to training DiT only
                
        print(f"🔧 Training Configuration:")
        print(f"   Trainable models: {trainable_models}")
        print(f"   VACE-E enabled: {enable_vace_e}")
        
        self.pipe.freeze_except(trainable_models.split(","))
        
        # Add LoRA to the base models (supports vace_e)
        if lora_base_model is not None:
            model = self.add_lora_to_model(
                getattr(self.pipe, lora_base_model),
                target_modules=lora_target_modules.split(","),
                lora_rank=lora_rank
            )
            setattr(self.pipe, lora_base_model, model)
            
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        
        # Debug: Show trainable parameters after configuration
        trainable_params = list(self.trainable_modules())
        param_count = sum(p.numel() for p in trainable_params)
        print(f"💡 Trainable Parameters:")
        print(f"   Total parameters: {param_count:,}")
        print(f"   Number of parameter tensors: {len(trainable_params)}")
        
        if len(trainable_params) == 0:
            print("❌ ERROR: No trainable parameters found!")
            print("   Available models in pipeline:")
            for name in dir(self.pipe):
                if not name.startswith('_') and hasattr(getattr(self.pipe, name), 'parameters'):
                    model = getattr(self.pipe, name)
                    if hasattr(model, 'parameters'):
                        total_params = sum(p.numel() for p in model.parameters())
                        print(f"     {name}: {total_params:,} parameters")
        
        # Training step counter for CLUB loss
        self.training_step = 0

    def forward_preprocess(self, data):
        # Determine batch size from video tensor
        batch_size = data["video"].shape[0] if "video" in data else 1
        print(f"DEBUG: Determined batch size: {batch_size}")
        print(f"DEBUG: Video shape: {data['video'].shape if 'video' in data else 'None'}")
        
        # CFG-sensitive parameters
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        
        # CFG-unsensitive parameters
        inputs_shared = {
            # Standard video training parameters (following train.py pattern)
            "input_video": data["video"],
            "height": data["video"].shape[3],
            "width": data["video"].shape[4],
            "num_frames": data["video"].shape[1],
            "batch_size": batch_size,  # Add batch size for pipeline units
            # Training-specific parameters (do not modify unless you know what this causes)
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
        }
        
        # Add standard VACE inputs if available
        if "vace_video" in data:
            inputs_shared["vace_video"] = data["vace_video"]
        if "vace_video_mask" in data:
            inputs_shared["vace_video_mask"] = data["vace_video_mask"]
        if "vace_reference_image" in data:
            inputs_shared["vace_reference_image"] = data["vace_reference_image"]
        
        # Add VACE-E inputs if available
        # 1. Text features
        if "prompt" in data:
            # Let the pipeline encode the prompt to get text features
            vace_e_text_features = self.pipe.prompter.encode_prompt(data["prompt"], device=self.pipe.device)
            inputs_shared["vace_e_text_features"] = vace_e_text_features
            inputs_shared["vace_e_text_mask"] = torch.ones(vace_e_text_features.shape[0], vace_e_text_features.shape[1], device=self.pipe.device).bool()
        
        # 2. Hand motion sequence  
        if "hand_motion_sequence" in data and data["hand_motion_sequence"] is not None:
            hand_motion = data["hand_motion_sequence"]
            # Add batch dimension if needed
            if hand_motion.dim() == 2:
                hand_motion = hand_motion.unsqueeze(0)
            inputs_shared["vace_e_hand_motion_sequence"] = hand_motion
            inputs_shared["vace_e_motion_mask"] = torch.ones(hand_motion.shape[0], hand_motion.shape[1], device=self.pipe.device).bool()
        
        # 3. Object trajectory sequence
        if "object_trajectory_sequence" in data and data["object_trajectory_sequence"] is not None:
            obj_traj = data["object_trajectory_sequence"]
            # Add batch dimension if needed
            if obj_traj.dim() == 3:
                obj_traj = obj_traj.unsqueeze(0)
            inputs_shared["vace_e_object_trajectory_sequence"] = obj_traj
            inputs_shared["vace_e_trajectory_mask"] = torch.ones(obj_traj.shape[0], obj_traj.shape[1], obj_traj.shape[2], device=self.pipe.device).bool()
        
        # 4. Object IDs
        if "object_ids" in data and data["object_ids"] is not None:
            obj_ids = data["object_ids"]
            # Add batch dimension if needed
            if obj_ids.dim() == 1:
                obj_ids = obj_ids.unsqueeze(0)
            inputs_shared["vace_e_object_ids"] = obj_ids
        
        # 5. Embodiment image features  
        if "embodiment_image" in data and data["embodiment_image"] is not None:
            # Encode embodiment image using CLIP
            embodiment_image = self.pipe.preprocess_image(data["embodiment_image"]).to(self.pipe.device)
            vace_e_embodiment_image_features = self.pipe.image_encoder.encode_image([embodiment_image])
            inputs_shared["vace_e_embodiment_image_features"] = vace_e_embodiment_image_features
        
        # Set VACE-E scale
        inputs_shared["vace_e_scale"] = 1.0
        
        # Extra inputs (following train.py pattern)
        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][:, 0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][:, -1]
            else:
                inputs_shared[extra_input] = data[extra_input]
        
        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}
    
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.forward_preprocess(data)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        
        # Pass current training step to enable CLUB loss scheduling
        loss, log_loss = self.pipe.training_loss(training_step=self.training_step, **models, **inputs)
        
        # Increment training step counter
        self.training_step += 1
        
        return loss, log_loss
    
    def set_accelerator(self, accelerator):
        """
        Set the accelerator instance for the pipeline to enable distributed feature gathering.
        
        Args:
            accelerator: Accelerator instance from the training loop
        """
        self.pipe.set_accelerator(accelerator)


if __name__ == "__main__":
    # Use the same argument parser as the original, but add VACE-E specific arguments
    parser = wan_parser()
    
    # Add VACE-E specific arguments
    parser.add_argument("--enable_vace_e", action="store_true", default=True, help="Enable VACE-E task-embodiment fusion")
    parser.add_argument("--vace_e_layers", type=str, default="0,5,10,15,20,25", help="Comma-separated DiT layer indices for VACE-E hints")
    parser.add_argument("--vace_e_task_processing", action="store_true", default=True, help="Enable VACE-E task feature processing")
    
    # Dataset-specific arguments
    parser.add_argument("--task_metadata_path", type=str, default="/home/zhiyuan/Codes/human-policy/data/ph2d_metadata.json", help="Path to task metadata JSON")
    parser.add_argument("--max_hand_motion_length", type=int, default=512, help="Maximum hand motion sequence length")
    parser.add_argument("--max_object_trajectory_length", type=int, default=512, help="Maximum object trajectory sequence length")
    parser.add_argument("--max_objects", type=int, default=10, help="Maximum number of objects per episode")
    parser.add_argument("--fallback_to_video_only", action="store_true", help="Fall back to video-only training when robot data unavailable")
    
    # CLUB loss arguments
    parser.add_argument("--club_lambda", type=float, default=1.0, help="Weight for CLUB loss in total loss")
    parser.add_argument("--club_update_freq", type=int, default=1, help="Update CLUB estimator every N training steps")
    parser.add_argument("--club_training_steps", type=int, default=5, help="Number of CLUB training steps per update")
    parser.add_argument("--club_lr", type=float, default=1e-3, help="Learning rate for CLUB optimizer")
    parser.add_argument("--enable_club_loss", action="store_true", default=True, help="Enable CLUB loss for mutual information minimization")
    parser.add_argument("--disable_club_loss", action="store_true", help="Disable CLUB loss (overrides enable_club_loss)")

    # CLIP contrastive loss arguments
    # Contrastive loss configuration
    parser.add_argument("--contrastive_temperature", type=float, default=0.07, help="Temperature parameter for contrastive loss")
    parser.add_argument("--task_contrastive_lambda", type=float, default=1.0, help="Weight for task contrastive loss")
    parser.add_argument("--embodiment_contrastive_lambda", type=float, default=1.0, help="Weight for embodiment contrastive loss")
    parser.add_argument("--enable_contrastive_loss", action="store_true", default=True, help="Enable contrastive loss")
    parser.add_argument("--disable_contrastive_loss", action="store_true", help="Disable contrastive loss (overrides enable_contrastive_loss)")

    parser.add_argument("--resume_path", type=str, default=None, help="Resume training from checkpoint")
    
    args = parser.parse_args()
    
    # Initialize Accelerator first so each process only sees its own GPU
    # This prevents memory imbalance where everything loads on GPU 0
    print("🚀 Initializing Accelerator for balanced GPU memory usage...")
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[
            # This ensures DDP can handle parameters that don't receive gradients
            # Critical for VACE-E components that may not always be used
            DistributedDataParallelKwargs(find_unused_parameters=True)
        ]
    )
    
    # Set device based on accelerator's assignment
    device = accelerator.device
    print(f"📍 Process {accelerator.process_index} assigned to device: {device}")
    
    # Enable appropriate defaults for CLUB training
    args = enable_club_training_defaults(args)
    
    # Debug: Show key training arguments
    print(f"🔍 Training Arguments:")
    print(f"   trainable_models: {args.trainable_models}")
    print(f"   lora_base_model: {args.lora_base_model}")
    print(f"   enable_vace_e: {args.enable_vace_e}")
    print(f"   vace_e_task_processing: {args.vace_e_task_processing}")
    
    # Parse VACE-E layers
    vace_e_layers = tuple(map(int, args.vace_e_layers.split(","))) if args.vace_e_layers else (0, 5, 10, 15, 20, 25)
    
    # Create dataset with VACE-E support
    dataset = create_training_dataset(args)
    # Create model with assigned device
    model = WanTrainingModuleE(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        # VACE-E parameters
        enable_vace_e=args.enable_vace_e,
        vace_e_layers=vace_e_layers,
        vace_e_task_processing=args.vace_e_task_processing,
        resume_path=args.resume_path,
    )
    
    # Set device on model for balanced loading
    model._device = device
    
    # Verify we have trainable parameters before creating optimizer
    trainable_params = list(model.trainable_modules())
    if len(trainable_params) == 0:
        print("❌ FATAL ERROR: No trainable parameters found!")
        print("   Check your --trainable_models argument.")
        print("   For VACE-E training, use: --trainable_models 'dit,vace_e'")
        print("   For standard training, use: --trainable_models 'dit'")
        exit(1)
    else:
        param_count = sum(p.numel() for p in trainable_params)
        print(f"✅ Found {param_count:,} trainable parameters across {len(trainable_params)} tensors")
    
    # Configure CLUB loss for mutual information minimization
    enable_club = args.enable_club_loss and not args.disable_club_loss
    print(f"\n🎯 CLUB Loss Configuration:")
    print(f"   Enabled: {enable_club}")
    if enable_club:
        print(f"   Lambda weight: {args.club_lambda}")
        print(f"   Update frequency: {args.club_update_freq}")
        print(f"   Training steps per update: {args.club_training_steps}")
        print(f"   Learning rate: {args.club_lr}")
    
    model.pipe.configure_club_loss(
        lambda_weight=args.club_lambda,
        update_freq=args.club_update_freq,
        training_steps=args.club_training_steps,
        club_lr=args.club_lr,
        enable=enable_club
    )
    
    # Configure contrastive loss
    enable_contrastive = args.enable_contrastive_loss and not args.disable_contrastive_loss
    print(f"\n🎯 Contrastive Loss Configuration:")
    print(f"   Enabled: {enable_contrastive}")
    if enable_contrastive:
        print(f"   Temperature: {args.contrastive_temperature}")
        print(f"   Task lambda weight: {args.task_contrastive_lambda}")
        print(f"   Embodiment lambda weight: {args.embodiment_contrastive_lambda}")
    
    model.pipe.configure_contrastive_loss(
        temperature=args.contrastive_temperature,
        task_lambda=args.task_contrastive_lambda,
        embodiment_lambda=args.embodiment_contrastive_lambda,
        enable=enable_contrastive
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_config=args
    )
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    launch_training_task(
        dataset, model, model_logger, optimizer, scheduler,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_video_collate=args.use_video_collate,
        video_min_value=args.video_min_value,
        video_max_value=args.video_max_value,
        accelerator=accelerator,  # Pass the early-initialized accelerator
    )
