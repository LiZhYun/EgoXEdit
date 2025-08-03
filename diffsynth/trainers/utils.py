import imageio, os, torch, warnings, torchvision, argparse, json, numpy as np
from peft import LoraConfig, inject_adapter_in_model
from PIL import Image
import pandas as pd
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    warnings.warn("wandb is not installed. Logging to wandb will be disabled.")



class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        data_file_keys=("image",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        repeat=1,
        args=None,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat
            
        self.base_path = base_path
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.repeat = repeat

        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True
            
        if metadata_path is None:
            print("No metadata. Trying to generate it.")
            metadata = self.generate_metadata(base_path)
            print(f"{len(metadata)} lines in metadata.")
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        else:
            metadata = pd.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]


    def generate_metadata(self, folder):
        image_list, prompt_list = [], []
        file_set = set(os.listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext_name = file_name.split(".")[-1].lower()
            file_base_name = file_name[:-len(file_ext_name)-1]
            if file_ext_name not in self.image_file_extension:
                continue
            prompt_file_name = file_base_name + ".txt"
            if prompt_file_name not in file_set:
                continue
            with open(os.path.join(folder, prompt_file_name), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            image_list.append(file_name)
            prompt_list.append(prompt)
        metadata = pd.DataFrame()
        metadata["image"] = image_list
        metadata["prompt"] = prompt_list
        return metadata
    
    
    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    
    def get_height_width(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def load_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        image = self.crop_and_resize(image, *self.get_height_width(image))
        return image
    
    
    def load_data(self, file_path):
        return self.load_image(file_path)


    def __getitem__(self, data_id):
        data = self.data[data_id % len(self.data)].copy()
        for key in self.data_file_keys:
            if key in data:
                path = os.path.join(self.base_path, data[key])
                data[key] = self.load_data(path)
                if data[key] is None:
                    warnings.warn(f"cannot load file {data[key]}.")
                    return None
        return data
    

    def __len__(self):
        return len(self.data) * self.repeat



class VideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        num_frames=81,
        time_division_factor=4, time_division_remainder=1,
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        data_file_keys=("video",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        video_file_extension=("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"),
        repeat=1,
        args=None,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            num_frames = args.num_frames
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat
        
        self.base_path = base_path
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.video_file_extension = video_file_extension
        self.repeat = repeat
        
        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True
            
        if metadata_path is None:
            print("No metadata. Trying to generate it.")
            metadata = self.generate_metadata(base_path)
            print(f"{len(metadata)} lines in metadata.")
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        else:
            metadata = pd.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
            
    
    def generate_metadata(self, folder):
        video_list, prompt_list = [], []
        file_set = set(os.listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext_name = file_name.split(".")[-1].lower()
            file_base_name = file_name[:-len(file_ext_name)-1]
            if file_ext_name not in self.image_file_extension and file_ext_name not in self.video_file_extension:
                continue
            prompt_file_name = file_base_name + ".txt"
            if prompt_file_name not in file_set:
                continue
            with open(os.path.join(folder, prompt_file_name), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            video_list.append(file_name)
            prompt_list.append(prompt)
        metadata = pd.DataFrame()
        metadata["video"] = video_list
        metadata["prompt"] = prompt_list
        return metadata
        
        
    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    
    def get_height_width(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def get_num_frames(self, reader):
        num_frames = self.num_frames
        if int(reader.count_frames()) < num_frames:
            num_frames = int(reader.count_frames())
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
    

    def load_video(self, file_path):
        reader = imageio.get_reader(file_path)
        num_frames = self.get_num_frames(reader)
        frames = []
        for frame_id in range(num_frames):
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame, *self.get_height_width(frame))
            frames.append(frame)
        reader.close()
        return frames
    
    
    def load_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        image = self.crop_and_resize(image, *self.get_height_width(image))
        frames = [image]
        return frames
    
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.image_file_extension
    
    
    def is_video(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.video_file_extension
    
    
    def load_data(self, file_path):
        if self.is_image(file_path):
            return self.load_image(file_path)
        elif self.is_video(file_path):
            return self.load_video(file_path)
        else:
            return None


    def __getitem__(self, data_id):
        data = self.data[data_id % len(self.data)].copy()
        for key in self.data_file_keys:
            if key in data:
                path = os.path.join(self.base_path, data[key])
                data[key] = self.load_data(path)
                if data[key] is None:
                    warnings.warn(f"cannot load file {data[key]}.")
                    return None
        return data
    

    def __len__(self):
        return len(self.data) * self.repeat



class DiffusionTrainingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        
    def to(self, *args, **kwargs):
        for name, model in self.named_children():
            model.to(*args, **kwargs)
        return self
        
        
    def trainable_modules(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.parameters())
        return trainable_modules
    
    
    def trainable_param_names(self):
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        return trainable_param_names
    
    
    def add_lora_to_model(self, model, target_modules, lora_rank, lora_alpha=None):
        if lora_alpha is None:
            lora_alpha = lora_rank
        lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules)
        model = inject_adapter_in_model(lora_config, model)
        return model
    
    
    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        trainable_param_names = self.trainable_param_names()
        state_dict = {name: param for name, param in state_dict.items() if name in trainable_param_names}
        if remove_prefix is not None:
            state_dict_ = {}
            # Support comma-separated multiple prefixes
            prefixes = [p.strip() for p in remove_prefix.split(',')] if ',' in remove_prefix else [remove_prefix]
            
            for name, param in state_dict.items():
                new_name = name
                # Try to remove any matching prefix
                for prefix in prefixes:
                    if name.startswith(prefix):
                        new_name = name[len(prefix):]
                        break
                state_dict_[new_name] = param
            state_dict = state_dict_
        return state_dict



class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x, 
                 use_wandb=False, wandb_project=None, wandb_config=None):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.use_wandb = use_wandb and HAS_WANDB
        self.step_count = 0
        
        if self.use_wandb:
            wandb.init(
                entity="zhiyuanli",
                project=wandb_project or "diffsynth-training",
                config=wandb_config
            )
        elif use_wandb and not HAS_WANDB:
            warnings.warn("wandb was requested but is not available. Install wandb to enable logging.")
        
    
    def on_step_end(self, log_loss):
        # log_loss = {
        #         'total_loss': total_loss,
        #         'flow_loss': flow_loss,
        #         'club_loss': club_loss
        #     }
        self.step_count += 1
        
        if self.use_wandb and log_loss:
            # Log all loss components to wandb
            wandb_log_dict = {}
            for key, value in log_loss.items():
                if torch.is_tensor(value):
                    wandb_log_dict[f"train/{key}"] = value.item()
                else:
                    wandb_log_dict[f"train/{key}"] = value
            
            wandb_log_dict["train/step"] = self.step_count
            wandb.log(wandb_log_dict, step=self.step_count)
    
    
    def on_epoch_end(self, accelerator, model, epoch_id):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)
            
            if self.use_wandb:
                # Log epoch completion
                wandb.log({
                    "train/epoch": epoch_id,
                    "train/checkpoint_saved": True
                }, step=self.step_count)
                
                # Optionally save model as wandb artifact
                artifact = wandb.Artifact(
                    name=f"model-epoch-{epoch_id}",
                    type="model",
                    description=f"Model checkpoint at epoch {epoch_id}"
                )
                artifact.add_file(path)
                wandb.log_artifact(artifact)
    
    
    def finish(self):
        """Clean up wandb run if active."""
        if self.use_wandb:
            wandb.finish()



def video_collate_fn(batch, min_value=-1, max_value=1):
    """
    Custom collate function for video datasets that handles batching of:
    - Videos (lists of PIL Images) -> torch.Tensor [B, T, C, H, W]
    - Hand motion sequences (tensors) -> torch.Tensor [B, seq_len, feature_dim] (padded)
    - Object trajectories (tensors) -> torch.Tensor [B, seq_len, num_objects, 9] (padded)
    - Object IDs (tensors) -> torch.Tensor [B, num_objects] (padded)
    - Images (PIL Images) -> torch.Tensor [B, C, H, W]
    - Strings and None values -> lists
    
    This collate function is essential for CLUB loss training which requires batch_size > 1
    to properly estimate mutual information between task and embodiment features.
    
    Args:
        batch: List of dataset items to collate
        min_value: Minimum value for pixel normalization (default: -1)
        max_value: Maximum value for pixel normalization (default: 1)
    
    Usage:
        dataloader = DataLoader(dataset, batch_size=4, collate_fn=video_collate_fn)
        # Or use launch_training_task with use_video_collate=True
        # With custom normalization:
        collate_fn = lambda batch: video_collate_fn(batch, min_value=0, max_value=1)
    """
    # Filter out None items
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        return None
    
    # Initialize result dictionary
    result = {}
    
    # Get all keys from the first item
    keys = batch[0].keys()
    
    for key in keys:
        values = [item[key] for item in batch]
        
        if key in ['prompt', 'task_name', 'episode_name']:
            # String data - keep as list
            result[key] = values
        elif key in ['video', 'vace_video', 'vace_video_mask']:
            # Video data (list of PIL Images) - need to handle None values
            valid_videos = [v for v in values if v is not None]
            if len(valid_videos) == 0:
                result[key] = None
            else:
                # Convert PIL Images to tensors and batch
                # Assuming all videos have same shape after preprocessing
                video_tensors = []
                for video in valid_videos:
                    if isinstance(video, list) and len(video) > 0:
                        # Convert PIL images to tensors
                        frames = []
                        for frame in video:
                            if frame is not None:
                                # Follow the exact same pattern as preprocess_image in wan_video_new_E.py
                                frame_tensor = torch.Tensor(np.array(frame, dtype=np.float32)) if isinstance(frame, Image.Image) else frame
                                frame_tensor = frame_tensor.to(dtype=torch.float32)
                                frame_tensor = frame_tensor * ((max_value - min_value) / 255) + min_value
                                # Convert from H W C to C H W
                                if len(frame_tensor.shape) == 3:
                                    frame_tensor = frame_tensor.permute(2, 0, 1)
                                frames.append(frame_tensor)
                        if len(frames) > 0:
                            video_tensor = torch.stack(frames)  # [T, C, H, W]
                            video_tensors.append(video_tensor)
                
                if len(video_tensors) > 0:
                    # Pad videos to same length if needed
                    max_frames = max(v.shape[0] for v in video_tensors)
                    padded_videos = []
                    for v in video_tensors:
                        if v.shape[0] < max_frames:
                            # Pad with last frame
                            last_frame = v[-1:].expand(max_frames - v.shape[0], -1, -1, -1)
                            v = torch.cat([v, last_frame], dim=0)
                        padded_videos.append(v)
                    result[key] = torch.stack(padded_videos)  # [B, T, C, H, W]
                else:
                    result[key] = None
        elif key in ['embodiment_image', 'vace_reference_image']:
            # Single image data
            valid_images = [v for v in values if v is not None]
            if len(valid_images) == 0:
                result[key] = None
            else:
                image_tensors = []
                for img in valid_images:
                    if img is not None:
                        # Follow the exact same pattern as preprocess_image in wan_video_new_E.py
                        img_tensor = torch.Tensor(np.array(img, dtype=np.float32)) if isinstance(img, Image.Image) else img
                        img_tensor = img_tensor.to(dtype=torch.float32)
                        img_tensor = img_tensor * ((max_value - min_value) / 255) + min_value
                        # Convert from H W C to C H W
                        if len(img_tensor.shape) == 3:
                            img_tensor = img_tensor.permute(2, 0, 1)
                        image_tensors.append(img_tensor)
                if len(image_tensors) > 0:
                    result[key] = torch.stack(image_tensors)  # [B, C, H, W]
                else:
                    result[key] = None
        elif key in ['hand_motion_sequence', 'object_trajectory_sequence', 'object_ids']:
            # Tensor data that needs padding
            valid_tensors = [v for v in values if v is not None]
            if len(valid_tensors) == 0:
                result[key] = None
            else:
                if len(valid_tensors) == 1:
                    result[key] = valid_tensors[0].unsqueeze(0)  # Add batch dimension
                else:
                    # Pad to same length
                    if key == 'object_trajectory_sequence':
                        # [seq_len, num_objects, 9] -> need to pad both seq_len and num_objects
                        max_seq_len = max(t.shape[0] for t in valid_tensors)
                        max_objects = max(t.shape[1] for t in valid_tensors)
                        padded_tensors = []
                        for t in valid_tensors:
                            # Pad sequence length
                            if t.shape[0] < max_seq_len:
                                pad_seq = torch.zeros(max_seq_len - t.shape[0], t.shape[1], t.shape[2])
                                t = torch.cat([t, pad_seq], dim=0)
                            # Pad number of objects
                            if t.shape[1] < max_objects:
                                pad_obj = torch.zeros(t.shape[0], max_objects - t.shape[1], t.shape[2])
                                t = torch.cat([t, pad_obj], dim=1)
                            padded_tensors.append(t)
                        result[key] = torch.stack(padded_tensors)
                    elif key == 'object_ids':
                        # [num_objects] -> pad to same number of objects
                        max_objects = max(t.shape[0] for t in valid_tensors)
                        padded_tensors = []
                        for t in valid_tensors:
                            if t.shape[0] < max_objects:
                                pad = torch.zeros(max_objects - t.shape[0], dtype=t.dtype)
                                t = torch.cat([t, pad], dim=0)
                            padded_tensors.append(t)
                        result[key] = torch.stack(padded_tensors)
                    else:
                        # hand_motion_sequence: [seq_len, feature_dim]
                        max_seq_len = max(t.shape[0] for t in valid_tensors)
                        padded_tensors = []
                        for t in valid_tensors:
                            if t.shape[0] < max_seq_len:
                                pad = torch.zeros(max_seq_len - t.shape[0], t.shape[1])
                                t = torch.cat([t, pad], dim=0)
                            padded_tensors.append(t)
                        result[key] = torch.stack(padded_tensors)
        else:
            # Default: keep as list
            result[key] = values
    
    return result


def launch_training_task(
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    num_epochs: int = 1,
    gradient_accumulation_steps: int = 1,
    batch_size: int = 1,
    num_workers: int = 0,
    use_video_collate: bool = False,
    video_min_value: float = -1,
    video_max_value: float = 1,
):
    # Choose appropriate collate function
    collate_fn = (lambda batch: video_collate_fn(batch, min_value=video_min_value, max_value=video_max_value)) if use_video_collate else (lambda x: x[0])
    
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        shuffle=True, 
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True  # Drop last incomplete batch for consistent CLUB training
    )
    # Configure DDP to handle unused parameters (e.g., VACE-E components that may not always be used)
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        # kwargs_handlers=[
        #     # This ensures DDP can handle parameters that don't receive gradients
        #     # Common when some model components are conditionally used
        #     DistributedDataParallelKwargs(find_unused_parameters=True)
        # ]
    )
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    
    for epoch_id in range(num_epochs):
        for data in tqdm(dataloader):
            if data is None:  # Skip None batches
                continue
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                loss, log_loss = model(data)
                accelerator.backward(loss)
                optimizer.step()
                model_logger.on_step_end(log_loss)
                scheduler.step()
        model_logger.on_epoch_end(accelerator, model, epoch_id)



def launch_data_process_task(model: DiffusionTrainingModule, dataset, output_path="./models"):
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0])
    accelerator = Accelerator()
    model, dataloader = accelerator.prepare(model, dataloader)
    os.makedirs(os.path.join(output_path, "data_cache"), exist_ok=True)
    for data_id, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            inputs = model.forward_preprocess(data)
            inputs = {key: inputs[key] for key in model.model_input_keys if key in inputs}
            torch.save(inputs, os.path.join(output_path, "data_cache", f"{data_id}.pth"))


def enable_club_training_defaults(args):
    """
    Helper function to set appropriate defaults for CLUB loss training.
    CLUB loss requires batch_size > 1 to compute mutual information properly.
    """
    if hasattr(args, 'enable_club_loss') and args.enable_club_loss:
        if args.batch_size <= 1:
            print("⚠️  Warning: CLUB loss requires batch_size > 1 for proper mutual information estimation.")
            print("   Automatically setting batch_size=4 and use_video_collate=True")
            args.batch_size = 2
            args.use_video_collate = True
        
        if not args.use_video_collate:
            print("⚠️  Warning: CLUB loss with video data requires use_video_collate=True for proper batching.")
            print("   Automatically enabling use_video_collate")
            args.use_video_collate = True
            
        print(f"✅ CLUB training configured: batch_size={args.batch_size}, use_video_collate={args.use_video_collate}")
    
    return args



def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset_base_path", type=str, default="", required=True, help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--max_pixels", type=int, default=1280*720, help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=None, help="Height of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames per video. Frames are sampled from the video prefix.")
    parser.add_argument("--data_file_keys", type=str, default="image,video", help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of dataloader workers.")
    parser.add_argument("--use_video_collate", default=True, action="store_true", help="Use video collate function for batching.")
    parser.add_argument("--video_min_value", type=float, default=-1, help="Minimum value for video pixel normalization in collate function.")
    parser.add_argument("--video_max_value", type=float, default=1, help="Maximum value for video pixel normalization in collate function.")
    parser.add_argument("--use_wandb", default=True, action="store_true", help="Whether to use wandb for logging.")
    parser.add_argument("--wandb_project", type=str, default="wanvideo-training", help="Wandb project name.")
    return parser



def flux_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset_base_path", type=str, default="", required=True, help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--max_pixels", type=int, default=1024*1024, help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=None, help="Height of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--data_file_keys", type=str, default="image", help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--align_to_opensource_format", default=False, action="store_true", help="Whether to align the lora format to opensource format. Only for DiT's LoRA.")
    parser.add_argument("--use_gradient_checkpointing", default=False, action="store_true", help="Whether to use gradient checkpointing.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of dataloader workers.")
    parser.add_argument("--use_video_collate", default=False, action="store_true", help="Use video collate function for batching.")
    parser.add_argument("--use_wandb", default=False, action="store_true", help="Whether to use wandb for logging.")
    parser.add_argument("--wandb_project", type=str, default=None, help="Wandb project name.")
    parser.add_argument("--wandb_name", type=str, default=None, help="Wandb run name.")
    return parser
