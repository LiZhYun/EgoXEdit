import torch
import os
import json
import h5py
import numpy as np
import warnings
import imageio
import torchvision
from PIL import Image
from tqdm import tqdm


class VideoDatasetE(torch.utils.data.Dataset):
    """
    Standalone dataset for VACE-E training with robot demonstration data.
    
    Designed specifically for the robotic data structure:
    - Base path: /home/zhiyuan/Codes/DataSets/small_test/
    - Task metadata: /home/zhiyuan/Codes/human-policy/data/ph2d_metadata.json
    - Episode structure: task_folder/episode_N/files
    
    Expected folder structure:
    base_path/
    ├── task_folder_1/
    │   ├── episode_0/
    │   │   ├── episode_0.mp4                       # Target video to generate
    │   │   ├── episode_0_hand_trajectories.hdf5    # Dual-hand motion data
    │   │   ├── episode_0_object_trajectories.hdf5  # Object trajectory data
    │   │   ├── episode_0_hands_masked.mp4          # VACE control video
    │   │   ├── episode_0_hands_mask.mp4            # VACE mask video
    │   │   └── end_effector_*.jpg                  # Embodiment images
    │   └── episode_1/...
    └── task_folder_2/...
    
    HDF5 Data Structure:
    hand_trajectories.hdf5:
        ├── left_wrist/
        │   ├── positions      # [frames, 3]
        │   └── rotations_6d   # [frames, 6]
        ├── right_wrist/
        │   ├── positions      # [frames, 3] 
        │   └── rotations_6d   # [frames, 6]
        ├── left_hand_states   # [frames] - gripper states
        └── right_hand_states  # [frames] - gripper states
    
    object_trajectories.hdf5:
        ├── object_0/
        │   ├── positions      # [frames, 3]
        │   ├── rotations_6d   # [frames, 6]
        │   └── attrs: object_id
        └── object_1/...
    
    IMPORTANT: Uses natural sequence lengths without padding!
    The model handles variable-length sequences with attention masks.
    
    Returns data format:
    {
        "video": List[PIL.Image],                    # Target video to generate
        "prompt": str,                               # Generated task prompt
        "vace_video": List[PIL.Image],               # VACE control video (hands masked)
        "vace_video_mask": List[PIL.Image],          # VACE mask video  
        "vace_reference_image": PIL.Image,           # End-effector reference
        "hand_motion_sequence": torch.Tensor,        # [natural_seq_len, 20] dual-hand motion
        "object_trajectory_sequence": torch.Tensor,  # [natural_seq_len, num_objects, 9]
        "object_ids": torch.Tensor,                  # [num_objects] object IDs
        "embodiment_image": PIL.Image,               # End-effector camera view
        "task_name": str,                           # Task identifier
        "episode_name": str,                        # Episode identifier
    }
    """
    
    def __init__(
        self,
        base_path="/home/zhiyuan/Codes/DataSets/small_test",
        task_metadata_path="/home/zhiyuan/Codes/human-policy/data/ph2d_metadata.json",
        num_frames=81,
        time_division_factor=4, 
        time_division_remainder=1,
        max_pixels=1920*1080, 
        height=None, 
        width=None,
        height_division_factor=16, 
        width_division_factor=16,
        max_hand_motion_length=512,
        max_object_trajectory_length=512,
        max_objects=10,
        repeat=1,
        enable_fallback=True,
        args=None,
    ):
        # Parse args if provided
        if args is not None:
            base_path = getattr(args, 'dataset_base_path', base_path)
            task_metadata_path = getattr(args, 'task_metadata_path', task_metadata_path)
            height = getattr(args, 'height', height)
            width = getattr(args, 'width', width)
            max_pixels = getattr(args, 'max_pixels', max_pixels)
            num_frames = getattr(args, 'num_frames', num_frames)
            repeat = getattr(args, 'dataset_repeat', repeat)
            max_hand_motion_length = getattr(args, 'max_hand_motion_length', max_hand_motion_length)
            max_object_trajectory_length = getattr(args, 'max_object_trajectory_length', max_object_trajectory_length)
            max_objects = getattr(args, 'max_objects', max_objects)
            enable_fallback = getattr(args, 'fallback_to_video_only', enable_fallback)
        
        self.base_path = base_path
        self.task_metadata_path = task_metadata_path
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.max_hand_motion_length = max_hand_motion_length
        self.max_object_trajectory_length = max_object_trajectory_length
        self.max_objects = max_objects
        self.repeat = repeat
        self.enable_fallback = enable_fallback
        
        # Set resolution mode
        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True
        
        # Load task metadata (ph2d format)
        self.task_metadata = {}
        if os.path.exists(task_metadata_path):
            try:
                with open(task_metadata_path, 'r') as f:
                    metadata_json = json.load(f)
                # Extract per_task_attributes from the JSON structure
                self.task_metadata = metadata_json.get("per_task_attributes", {})
                print(f"✓ Loaded robot task metadata: {len(self.task_metadata)} tasks")
                
                # Print some example task IDs for debugging
                example_tasks = list(self.task_metadata.keys())[:3]
                print(f"   Example task IDs: {example_tasks}")
            except Exception as e:
                warnings.warn(f"Failed to load task metadata from {task_metadata_path}: {e}")
        else:
            warnings.warn(f"Task metadata file not found: {task_metadata_path}")
        
        # Discover episodes from folder structure
        print(f"Scanning robot demonstration data from: {base_path}")
        self.episodes = self._discover_episodes()
        print(f"✓ Found {len(self.episodes)} episodes across {len(set(ep['task_name'] for ep in self.episodes))} tasks")
        
        # Validate episodes
        valid_episodes = []
        print("Validating episodes...")
        for episode in tqdm(self.episodes, desc="Validating"):
            if self._validate_episode(episode):
                valid_episodes.append(episode)
        
        self.episodes = valid_episodes
        print(f"✓ {len(self.episodes)} episodes passed validation")
        
        if len(self.episodes) == 0:
            raise ValueError(f"No valid episodes found in {base_path}")
    
    def _discover_episodes(self):
        """Discover all episodes from the folder structure."""
        episodes = []
        
        if not os.path.exists(self.base_path):
            warnings.warn(f"Base path does not exist: {self.base_path}")
            return episodes
        
        # Scan task folders
        for task_folder in os.listdir(self.base_path):
            task_path = os.path.join(self.base_path, task_folder)
            if not os.path.isdir(task_path):
                continue
            
            # Use full task folder name as the task identifier (matches ph2d_metadata.json keys)
            task_name = task_folder
            
            # Scan episode folders within task
            for episode_folder in os.listdir(task_path):
                episode_path = os.path.join(task_path, episode_folder)
                if not os.path.isdir(episode_path) or not episode_folder.startswith('episode_'):
                    continue
                
                episode_info = {
                    'task_name': task_name,           # Full folder name for metadata lookup
                    'task_folder': task_folder,       # Same as task_name
                    'episode_name': episode_folder,
                    'episode_path': episode_path,
                }
                episodes.append(episode_info)
        
        return episodes
    
    def _validate_episode(self, episode):
        """Validate that an episode has required files."""
        episode_path = episode['episode_path']
        episode_name = episode['episode_name']
        
        # Required files
        required_files = [
            f"{episode_name}.mp4",                     # Target video to generate
            f"{episode_name}_hands_masked.mp4",        # VACE control video  
            f"{episode_name}_hands_mask.mp4",          # VACE mask video
        ]
        
        # Check required files exist
        missing_files = []
        for required_file in required_files:
            file_path = os.path.join(episode_path, required_file)
            if not os.path.exists(file_path):
                missing_files.append(required_file)
        
        if missing_files:
            if not self.enable_fallback:
                return False
            warnings.warn(f"Episode {episode_name} missing files: {missing_files}")
        
        return True
    
    def generate_task_prompt(self, task_name, full_task_name=None):
        """Generate task-specific prompt from ph2d metadata."""
        # task_name should be the full folder name that matches ph2d_metadata.json keys
        metadata = self.task_metadata.get(task_name)
        
        if metadata is None:
            return f"Robot manipulation task: {task_name}"
        
        task_type = metadata.get("task_type", "manipulation")
        objects = metadata.get("objects", "objects")
        left_hand = metadata.get("left_hand", False)
        right_hand = metadata.get("right_hand", True)
        
        # Generate prompt based on hand usage
        if left_hand and right_hand:
            if task_type == "pouring":
                prompt = f"Pouring, left hand cup, right hand bottle"
            else:
                prompt = f"Both hands {task_type} {objects}"
        elif right_hand:
            prompt = f"Right hand {task_type} {objects}"
        elif left_hand:
            prompt = f"Left hand {task_type} {objects}"
        else:
            prompt = f"{task_type.capitalize()} {objects}"
        
        return prompt
    
    def crop_and_resize(self, image, target_height, target_width):
        """Crop and resize image to target dimensions."""
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
        """Get target height and width for image."""
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
        """Get number of frames to load from video."""
        num_frames = self.num_frames
        if int(reader.count_frames()) < num_frames:
            num_frames = int(reader.count_frames())
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
    
    def load_video(self, file_path):
        """Load video frames as list of PIL images."""
        try:
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
        except Exception as e:
            warnings.warn(f"Failed to load video {file_path}: {e}")
            return None
    
    def load_image(self, file_path):
        """Load single image as PIL Image."""
        try:
            image = Image.open(file_path).convert("RGB")
            image = self.crop_and_resize(image, *self.get_height_width(image))
            return image
        except Exception as e:
            warnings.warn(f"Failed to load image {file_path}: {e}")
            return None
    
    def load_hand_motion_from_hdf5(self, hdf5_path):
        """Load hand motion sequence from HDF5 and convert to 20D dual-hand format."""
        try:
            with h5py.File(hdf5_path, 'r') as f:
                # Try dual-hand format first
                if 'left_wrist' in f and 'right_wrist' in f:
                    # Load left hand data
                    left_positions = f['left_wrist/positions'][:]     # [frames, 3]
                    left_rotations = f['left_wrist/rotations_6d'][:]  # [frames, 6]
                    
                    # Load right hand data  
                    right_positions = f['right_wrist/positions'][:]     # [frames, 3]
                    right_rotations = f['right_wrist/rotations_6d'][:]  # [frames, 6]
                    
                    # Load gripper states for both hands
                    left_gripper_states = np.zeros(len(left_positions))  # Default: closed
                    right_gripper_states = np.zeros(len(right_positions))  # Default: closed
                    
                    if 'left_hand_states' in f:
                        left_gripper_states = f['left_hand_states'][:]
                        # Convert to binary: closed=0, open=1
                        left_gripper_states = (left_gripper_states == 1).astype(float)
                    
                    if 'right_hand_states' in f:
                        right_gripper_states = f['right_hand_states'][:]
                        # Convert to binary: closed=0, open=1  
                        right_gripper_states = (right_gripper_states == 1).astype(float)
                    
                    # Combine into 20D: [left_wrist(9), right_wrist(9), left_gripper(1), right_gripper(1)]
                    left_wrist_9d = np.concatenate([left_positions, left_rotations], axis=1)   # [frames, 9]
                    right_wrist_9d = np.concatenate([right_positions, right_rotations], axis=1) # [frames, 9]
                    
                    hand_motion = np.concatenate([
                        left_wrist_9d,                           # [frames, 9] - left wrist pose
                        right_wrist_9d,                          # [frames, 9] - right wrist pose  
                        left_gripper_states.reshape(-1, 1),     # [frames, 1] - left gripper state
                        right_gripper_states.reshape(-1, 1)     # [frames, 1] - right gripper state
                    ], axis=1)  # Result: [frames, 20]
                
                # Fallback to single-hand format (only right wrist)
                elif 'right_wrist' in f:
                    right_positions = f['right_wrist/positions'][:]     # [frames, 3]
                    right_rotations = f['right_wrist/rotations_6d'][:]  # [frames, 6]
                    
                    # Load right gripper states
                    if 'right_hand_states' in f:
                        right_gripper_states = f['right_hand_states'][:]
                        right_gripper_states = (right_gripper_states == 1).astype(float)
                    else:
                        right_gripper_states = np.ones(len(right_positions)) * 0.5  # Neutral
                    
                    # Create dual-hand with left hand as zeros
                    left_wrist_zeros = np.zeros((len(right_positions), 9))    # Left wrist: all zeros
                    left_gripper_zeros = np.zeros(len(right_positions))       # Left gripper: closed
                    right_wrist_9d = np.concatenate([right_positions, right_rotations], axis=1)
                    
                    hand_motion = np.concatenate([
                        left_wrist_zeros,                        # [frames, 9] - left wrist (zeros)
                        right_wrist_9d,                          # [frames, 9] - right wrist (active)
                        left_gripper_zeros.reshape(-1, 1),      # [frames, 1] - left gripper (closed)
                        right_gripper_states.reshape(-1, 1)     # [frames, 1] - right gripper (active)
                    ], axis=1)  # Result: [frames, 20]
                
                else:
                    warnings.warn(f"No recognized hand motion data in {hdf5_path}")
                    return None
                
                # Use natural sequence length - no padding needed!
                # Model handles variable lengths with attention masks
                return torch.from_numpy(hand_motion).to(torch.bfloat16)
                
        except Exception as e:
            warnings.warn(f"Failed to load hand motion from {hdf5_path}: {e}")
            return None
    
    def load_object_trajectory_from_hdf5(self, hdf5_path):
        """Load object trajectory sequence from HDF5."""
        try:
            with h5py.File(hdf5_path, 'r') as f:
                # Get all object groups
                object_groups = [key for key in f.keys() if key.startswith('object_')]
                
                if not object_groups:
                    return None, None
                
                all_object_trajectories = []
                object_ids = []
                
                for obj_group_name in sorted(object_groups):
                    obj_group = f[obj_group_name]
                    obj_id = obj_group.attrs.get('object_id', len(object_ids))
                    
                    if 'positions' in obj_group and 'rotations_6d' in obj_group:
                        positions = obj_group['positions'][:]      # [frames, 3]
                        rotations_6d = obj_group['rotations_6d'][:] # [frames, 6]
                        
                        # Combine position + rotation = 9D per object
                        obj_trajectory = np.concatenate([positions, rotations_6d], axis=1)  # [frames, 9]
                        all_object_trajectories.append(obj_trajectory)
                        object_ids.append(obj_id)
                
                if not all_object_trajectories:
                    return None, None
                
                # Use natural sequence lengths - no padding needed!
                # Stack: [seq_len, num_objects, 9] where seq_len varies naturally
                trajectories_array = np.stack(all_object_trajectories, axis=1) if all_object_trajectories else None
                object_ids_array = np.array(object_ids) if object_ids else None
                
                return (
                    torch.from_numpy(trajectories_array).to(torch.bfloat16) if trajectories_array is not None else None,
                    torch.from_numpy(object_ids_array).long() if object_ids_array is not None else None
                )
                
        except Exception as e:
            warnings.warn(f"Failed to load object trajectories from {hdf5_path}: {e}")
            return None, None
    
    def find_end_effector_image(self, episode_path):
        """Find end-effector image in episode folder."""
        for file_name in os.listdir(episode_path):
            if any(keyword in file_name.lower() for keyword in ['gripper', 'end_effector', 'robot', 'hand']) and file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                return os.path.join(episode_path, file_name)
        return None
    
    def __getitem__(self, data_id):
        """Load episode data with all modalities."""
        episode = self.episodes[data_id % len(self.episodes)]
        episode_path = episode['episode_path']
        episode_name = episode['episode_name']
        task_name = episode['task_name']
        
        data = {
            'task_name': task_name,
            'episode_name': episode_name,
            'prompt': self.generate_task_prompt(task_name),
        }
        
        # Load videos with correct distinction
        target_video_path = os.path.join(episode_path, f"{episode_name}.mp4")            # Target video to generate
        vace_video_path = os.path.join(episode_path, f"{episode_name}_hands_masked.mp4") # VACE control video
        vace_mask_path = os.path.join(episode_path, f"{episode_name}_hands_mask.mp4")    # VACE mask video
        
        # Load target video (what we want to generate)
        if os.path.exists(target_video_path):
            data['video'] = self.load_video(target_video_path)
        else:
            warnings.warn(f"Target video not found: {target_video_path}")
            if not self.enable_fallback:
                return None
            data['video'] = None
        
        # Load VACE control video (hands masked)
        if os.path.exists(vace_video_path):
            data['vace_video'] = self.load_video(vace_video_path)
        else:
            warnings.warn(f"VACE control video not found: {vace_video_path}")
            data['vace_video'] = None
        
        # Load VACE mask video
        if os.path.exists(vace_mask_path):
            data['vace_video_mask'] = self.load_video(vace_mask_path)
        else:
            warnings.warn(f"VACE mask not found: {vace_mask_path}")
            data['vace_video_mask'] = None
        
        # Load HDF5 robot data (optional)
        hand_hdf5_path = os.path.join(episode_path, f"{episode_name}_hand_trajectories.hdf5")
        if os.path.exists(hand_hdf5_path):
            data['hand_motion_sequence'] = self.load_hand_motion_from_hdf5(hand_hdf5_path)
        else:
            data['hand_motion_sequence'] = None
        
        obj_hdf5_path = os.path.join(episode_path, f"{episode_name}_object_trajectories.hdf5")
        if os.path.exists(obj_hdf5_path):
            obj_traj, obj_ids = self.load_object_trajectory_from_hdf5(obj_hdf5_path)
            data['object_trajectory_sequence'] = obj_traj
            data['object_ids'] = obj_ids
        else:
            data['object_trajectory_sequence'] = None
            data['object_ids'] = None
        
        # Load end-effector image (optional)
        end_effector_path = self.find_end_effector_image(episode_path)
        if end_effector_path:
            data['embodiment_image'] = self.load_image(end_effector_path)
            data['vace_reference_image'] = data['embodiment_image']  # Use as VACE reference
        else:
            data['embodiment_image'] = None
            data['vace_reference_image'] = None
        
        # Validate critical data
        if data['video'] is None:
            if not self.enable_fallback:
                return None
            warnings.warn(f"Episode {episode_name} missing critical video data")
        
        return data
    
    def __len__(self):
        """Dataset length with repeat factor."""
        return len(self.episodes) * self.repeat


def create_training_dataset(args):
    """Factory function to create VideoDatasetE for training."""
    return VideoDatasetE(args=args) 