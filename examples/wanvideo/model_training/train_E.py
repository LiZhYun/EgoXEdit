import torch, os, json
from diffsynth.pipelines.wan_video_new_E import WanVideoPipeline, ModelConfig
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, launch_training_task, wan_parser
from dataset_E import VideoDatasetE, create_training_dataset
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
        
        # Initialize VACE-E enhanced pipeline
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16, 
            device="cuda", 
            model_configs=model_configs,
            # VACE-E configuration
            enable_vace_e=enable_vace_e,
            vace_e_layers=vace_e_layers,
            vace_e_task_processing=vace_e_task_processing,
        )
        
        # Reset training scheduler
        self.pipe.scheduler.set_timesteps(1000, training=True)
        
        # Freeze untrainable models (include vace_e in potential trainable models)
        self.pipe.freeze_except([] if trainable_models is None else trainable_models.split(","))
        
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
        
        # Training step counter for CLUB loss
        self.training_step = 0

    def forward_preprocess(self, data):
        # CFG-sensitive parameters
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        
        # CFG-unsensitive parameters
        inputs_shared = {
            # Standard video training parameters (following train.py pattern)
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
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
            inputs_shared["vace_e_text_mask"] = torch.ones(1, vace_e_text_features.shape[1], device=self.pipe.device).bool()
        
        # 2. Hand motion sequence  
        if "hand_motion_sequence" in data and data["hand_motion_sequence"] is not None:
            hand_motion = data["hand_motion_sequence"]
            # Add batch dimension if needed
            if hand_motion.dim() == 2:
                hand_motion = hand_motion.unsqueeze(0)
            inputs_shared["vace_e_hand_motion_sequence"] = hand_motion
            inputs_shared["vace_e_motion_mask"] = torch.ones(1, hand_motion.shape[1], device=self.pipe.device).bool()
        
        # 3. Object trajectory sequence
        if "object_trajectory_sequence" in data and data["object_trajectory_sequence"] is not None:
            obj_traj = data["object_trajectory_sequence"]
            # Add batch dimension if needed
            if obj_traj.dim() == 3:
                obj_traj = obj_traj.unsqueeze(0)
            inputs_shared["vace_e_object_trajectory_sequence"] = obj_traj
            inputs_shared["vace_e_trajectory_mask"] = torch.ones(1, obj_traj.shape[1], obj_traj.shape[2], device=self.pipe.device).bool()
        
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
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
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
    
    args = parser.parse_args()
    
    # Parse VACE-E layers
    vace_e_layers = tuple(map(int, args.vace_e_layers.split(","))) if args.vace_e_layers else (0, 5, 10, 15, 20, 25)
    
    # Create dataset with VACE-E support
    dataset = create_training_dataset(args)
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
    )
    
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
    )
