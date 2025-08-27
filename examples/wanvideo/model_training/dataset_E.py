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
        
        # Validate episodes with detailed statistics
        total_episodes = len(self.episodes)
        valid_episodes = []
        filtered_count = 0
        validation_stats = {}
        
        print("Validating episodes with comprehensive checks...")
        for episode in tqdm(self.episodes, desc="Validating"):
            if self._validate_episode(episode):
                valid_episodes.append(episode)
            else:
                filtered_count += 1
                # Track validation failure reasons could be added here if needed
        
        self.episodes = valid_episodes
        
        # Print detailed statistics
        print(f"📊 Episode Validation Statistics:")
        print(f"   Total episodes found: {total_episodes}")
        print(f"   ✅ Valid episodes: {len(self.episodes)}")
        print(f"   ❌ Filtered episodes: {filtered_count}")
        if total_episodes > 0:
            print(f"   📈 Filtering rate: {filtered_count / total_episodes * 100:.1f}%")
            print(f"   📈 Retention rate: {len(self.episodes) / total_episodes * 100:.1f}%")
        
        if len(self.episodes) == 0:
            raise ValueError(f"No valid episodes found in {base_path} after filtering! Check your data and validation criteria.")
        elif len(self.episodes) < total_episodes * 0.5:
            warnings.warn(f"More than 50% of episodes were filtered out! Only {len(self.episodes)}/{total_episodes} episodes remain.")
        
        # Create consistent label mappings for CLIP contrastive loss
        self._create_label_mappings()
    
    def _create_label_mappings(self):
        """
        Create consistent label mappings for task prompts and embodiment types.
        This ensures the same task/embodiment gets the same label across all batches.
        """
        # Collect all unique task prompts and embodiment types
        unique_task_prompts = set()
        unique_embodiment_types = set()
        
        for episode in self.episodes:
            # Generate task prompt for this episode
            task_prompt = self.generate_task_prompt(episode['task_name'], episode.get('full_task_name'))
            unique_task_prompts.add(task_prompt)
            
            # Extract embodiment type from task name
            task_name = episode['task_name']
            if task_name.startswith('1'):
                embodiment_type = 'human'
            else:
                embodiment_type = 'robot'
            unique_embodiment_types.add(embodiment_type)
        
        # Create consistent mappings
        self.task_prompt_to_label = {prompt: idx for idx, prompt in enumerate(sorted(unique_task_prompts))}
        self.embodiment_type_to_label = {etype: idx for idx, etype in enumerate(sorted(unique_embodiment_types))}
        
        print(f"📊 CLIP Label Mappings Created:")
        print(f"   Unique task prompts: {len(self.task_prompt_to_label)}")
        print(f"   Task prompt samples: {list(self.task_prompt_to_label.keys())[:3]}")
        print(f"   Unique embodiment types: {len(self.embodiment_type_to_label)}")
        print(f"   Embodiment type mapping: {self.embodiment_type_to_label}")
    
    def get_task_label(self, task_prompt):
        """Get consistent task label for a given task prompt."""
        return self.task_prompt_to_label.get(task_prompt, 0)  # Default to 0 if not found
    
    def get_embodiment_label(self, task_name):
        """Get consistent embodiment label for a given task name."""
        if task_name.startswith('1'):
            embodiment_type = 'human'
        else:
            embodiment_type = 'robot'
        return self.embodiment_type_to_label.get(embodiment_type, 0)  # Default to 0 if not found
    
    def _validate_video_file(self, video_path, episode_name, video_type="Video"):
        """Comprehensive video file validation using multiple methods."""
        import subprocess
        import json
        
        # Method 1: ffprobe validation with timeout
        def validate_with_ffprobe(video_path, timeout=10):
            try:
                cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', video_path]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
                if result.returncode != 0:
                    return False, f"ffprobe failed with return code {result.returncode}"
                
                # Parse metadata
                data = json.loads(result.stdout)
                if 'streams' not in data or len(data['streams']) == 0:
                    return False, "No streams found in video"
                
                # Check for video stream
                video_streams = [s for s in data['streams'] if s.get('codec_type') == 'video']
                if not video_streams:
                    return False, "No video stream found"
                
                # Check video stream properties
                video_stream = video_streams[0]
                if video_stream.get('width', 0) <= 0 or video_stream.get('height', 0) <= 0:
                    return False, f"Invalid dimensions: {video_stream.get('width')}x{video_stream.get('height')}"
                
                # Check duration
                duration = float(video_stream.get('duration', 0))
                if duration <= 0:
                    return False, f"Invalid duration: {duration}"
                
                return True, "Valid video metadata"
                
            except subprocess.TimeoutExpired:
                return False, "ffprobe validation timeout"
            except json.JSONDecodeError:
                return False, "Invalid ffprobe output"
            except Exception as e:
                return False, f"ffprobe error: {e}"
        
        # Method 2: OpenCV validation
        def validate_with_opencv(video_path):
            try:
                import cv2
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    return False, "Cannot open with OpenCV"
                
                ret, frame = cap.read()
                if not ret or frame is None:
                    cap.release()
                    return False, "Cannot read first frame"
                
                # Check frame dimensions
                if frame.shape[0] == 0 or frame.shape[1] == 0:
                    cap.release()
                    return False, "Invalid frame dimensions"
                
                cap.release()
                return True, "OpenCV validation passed"
                
            except Exception as e:
                return False, f"OpenCV error: {e}"
        
        # Method 3: imageio validation (similar to actual loading)
        def validate_with_imageio(video_path):
            try:
                import imageio
                reader = imageio.get_reader(video_path, format="ffmpeg")
                
                # Try to get metadata
                meta = reader.get_meta_data()
                if not meta:
                    reader.close()
                    return False, "No metadata available"
                
                # Try to read first frame
                try:
                    first_frame = reader.get_data(0)
                    if first_frame is None or first_frame.size == 0:
                        reader.close()
                        return False, "First frame is empty"
                except Exception as e:
                    reader.close()
                    return False, f"Cannot read first frame: {e}"
                
                reader.close()
                return True, "imageio validation passed"
                
            except Exception as e:
                return False, f"imageio error: {e}"
        
        # Run all validations
        validations = [
            ("opencv", validate_with_opencv), 
            ("imageio", validate_with_imageio)
        ]
        
        for method_name, validate_fn in validations:
            try:
                if method_name == "ffprobe":
                    is_valid, reason = validate_fn(video_path)
                else:
                    is_valid, reason = validate_fn(video_path)
                
                if not is_valid:
                    print(f"❌ Episode {episode_name}: {video_type} video failed {method_name} validation: {reason}")
                    return False
                    
            except Exception as e:
                print(f"❌ Episode {episode_name}: {video_type} video validation error with {method_name}: {e}")
                return False
        
        return True

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
        """Validate episode metadata and file existence with comprehensive checks."""
        episode_path = episode['episode_path']
        episode_name = episode['episode_name']
        
        # Check if episode directory exists
        if not os.path.exists(episode_path):
            print(f"❌ Episode {episode_name}: Directory does not exist: {episode_path}")
            return False
        
        # Check if task name is valid
        if not episode.get('task_name'):
            print(f"❌ Episode {episode_name}: Missing task name")
            return False
        
        # Check target video (required)
        target_video_path = os.path.join(episode_path, f"{episode_name}.mp4")
        if not os.path.exists(target_video_path):
            print(f"❌ Episode {episode_name}: Target video not found: {target_video_path}")
            return False
        
        # Check video file size and integrity
        try:
            file_size = os.path.getsize(target_video_path)
            if file_size == 0:
                print(f"❌ Episode {episode_name}: Target video file is empty")
                return False
            if file_size < 1024:  # Less than 1KB is suspicious
                print(f"❌ Episode {episode_name}: Target video file is too small ({file_size} bytes)")
                return False
        except OSError:
            print(f"❌ Episode {episode_name}: Cannot access target video file")
            return False
        
        # Enhanced video validation with multiple methods
        if not self._validate_video_file(target_video_path, episode_name, "Target"):
            return False
        
        # Check VACE files with comprehensive validation
        vace_video_path = os.path.join(episode_path, f"{episode_name}_hands_masked.mp4")
        if os.path.exists(vace_video_path):
            try:
                file_size = os.path.getsize(vace_video_path)
                if file_size == 0:
                    print(f"❌ Episode {episode_name}: VACE video file is empty")
                    return False
                if file_size < 1024:
                    print(f"❌ Episode {episode_name}: VACE video file is too small ({file_size} bytes)")
                    return False
                
                # Comprehensive VACE video validation
                if not self._validate_video_file(vace_video_path, episode_name, "VACE"):
                    return False
                    
            except OSError:
                print(f"❌ Episode {episode_name}: Cannot access VACE video file")
                return False
        else:
            print(f"❌ Episode {episode_name}: VACE video file not found: {vace_video_path}")
            return False
        
        vace_mask_path = os.path.join(episode_path, f"{episode_name}_hands_mask.mp4")
        if os.path.exists(vace_mask_path):
            try:
                file_size = os.path.getsize(vace_mask_path)
                if file_size == 0:
                    print(f"❌ Episode {episode_name}: VACE mask file is empty")
                    return False
                if file_size < 1024:
                    print(f"❌ Episode {episode_name}: VACE mask file is too small ({file_size} bytes)")
                    return False
                
                # Comprehensive VACE mask validation
                if not self._validate_video_file(vace_mask_path, episode_name, "VACE Mask"):
                    return False
                    
            except OSError:
                print(f"❌ Episode {episode_name}: Cannot access VACE mask file")
                return False
        else:
            print(f"❌ Episode {episode_name}: VACE mask file not found: {vace_mask_path}")
            return False
        
        # Check HDF5 files if they exist
        hand_hdf5_path = os.path.join(episode_path, f"{episode_name}_hand_trajectories.hdf5")
        if os.path.exists(hand_hdf5_path):
            try:
                file_size = os.path.getsize(hand_hdf5_path)
                if file_size == 0:
                    print(f"❌ Episode {episode_name}: Hand motion HDF5 file is empty")
                    return False
                if file_size < 100:  # HDF5 files should be at least 100 bytes
                    print(f"❌ Episode {episode_name}: Hand motion HDF5 file is too small ({file_size} bytes)")
                    return False
            except OSError:
                print(f"❌ Episode {episode_name}: Cannot access hand motion HDF5 file")
                return False
        else:
            print(f"❌ Episode {episode_name}: Hand motion HDF5 file not found: {hand_hdf5_path}")
            return False
        
        obj_hdf5_path = os.path.join(episode_path, f"{episode_name}_object_trajectories.hdf5")
        if os.path.exists(obj_hdf5_path):
            try:
                file_size = os.path.getsize(obj_hdf5_path)
                if file_size == 0:
                    print(f"❌ Episode {episode_name}: Object trajectory HDF5 file is empty")
                    return False
                if file_size < 100:  # HDF5 files should be at least 100 bytes
                    print(f"❌ Episode {episode_name}: Object trajectory HDF5 file is too small ({file_size} bytes)")
                    return False
            except OSError:
                print(f"❌ Episode {episode_name}: Cannot access object trajectory HDF5 file")
                return False
        else:
            print(f"❌ Episode {episode_name}: Object trajectory HDF5 file not found: {obj_hdf5_path}")
            return False
        
        print(f"✅ Episode {episode_name}: Passed file validation")
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
            reader = imageio.get_reader(file_path, format="ffmpeg")
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
                
                # Validate the hand motion data
                if np.isnan(hand_motion).any() or np.isinf(hand_motion).any():
                    print(f"⚠️ Hand motion data contains NaN or infinite values. Replacing with zeros.")
                    hand_motion = np.nan_to_num(hand_motion, nan=0.0, posinf=0.0, neginf=0.0)
                
                print(f"✅ Loaded hand motion with shape: {hand_motion.shape}")
                return torch.from_numpy(hand_motion).to(torch.bfloat16)
                
        except Exception as e:
            warnings.warn(f"Failed to load hand motion from {hdf5_path}: {e}")
            return None
    
    def load_object_trajectory_from_hdf5(self, hdf5_path):
        """Load object trajectory sequence from HDF5 with proper padding."""
        try:
            with h5py.File(hdf5_path, 'r') as f:
                # Get all object groups
                object_groups = [key for key in f.keys() if key.startswith('object_')]
                
                if not object_groups:
                    return None, None
                
                all_object_trajectories = []
                object_ids = []
                trajectory_lengths = []
                
                # First pass: collect all trajectories and their lengths
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
                        trajectory_lengths.append(obj_trajectory.shape[0])
                        print(f"   Object {obj_id}: trajectory shape {obj_trajectory.shape}")
                
                if not all_object_trajectories:
                    return None, None
                
                # Find the maximum sequence length for padding
                max_seq_len = max(trajectory_lengths)
                min_seq_len = min(trajectory_lengths)
                
                if max_seq_len != min_seq_len:
                    print(f"⚠️ Object trajectories have different lengths: min={min_seq_len}, max={max_seq_len}. Padding to max length.")
                
                # Pad all trajectories to the same length
                padded_trajectories = []
                for i, trajectory in enumerate(all_object_trajectories):
                    current_len = trajectory.shape[0]
                    
                    if current_len < max_seq_len:
                        # Pad with the last frame (repeat the last frame)
                        last_frame = trajectory[-1:].copy()  # [1, 9]
                        padding_frames = max_seq_len - current_len
                        padding = np.tile(last_frame, (padding_frames, 1))  # [padding_frames, 9]
                        padded_trajectory = np.concatenate([trajectory, padding], axis=0)  # [max_seq_len, 9]
                    else:
                        padded_trajectory = trajectory
                    
                    padded_trajectories.append(padded_trajectory)
                
                # Stack all padded trajectories: [seq_len, num_objects, 9]
                trajectories_array = np.stack(padded_trajectories, axis=1)
                object_ids_array = np.array(object_ids)
                
                # Check if we have too many objects
                if len(object_ids) > self.max_objects:
                    print(f"⚠️ Too many objects ({len(object_ids)} > {self.max_objects}). Truncating to first {self.max_objects} objects.")
                    trajectories_array = trajectories_array[:, :self.max_objects, :]  # [seq_len, max_objects, 9]
                    object_ids_array = object_ids_array[:self.max_objects]  # [max_objects]
                
                print(f"✅ Loaded {len(object_ids_array)} objects with trajectory shape: {trajectories_array.shape}")
                
                # Validate the trajectory data
                if np.isnan(trajectories_array).any() or np.isinf(trajectories_array).any():
                    print(f"⚠️ Trajectory data contains NaN or infinite values. Replacing with zeros.")
                    trajectories_array = np.nan_to_num(trajectories_array, nan=0.0, posinf=0.0, neginf=0.0)
                
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
    
    def _validate_tensor_shape(self, tensor, expected_shape, tensor_name):
        """Validate tensor shape and return True if valid, False otherwise."""
        if tensor is None:
            return True  # None is acceptable for optional features
        
        if not isinstance(tensor, torch.Tensor):
            print(f"❌ {tensor_name}: Not a tensor, got {type(tensor)}")
            return False
        
        if len(tensor.shape) != len(expected_shape):
            print(f"❌ {tensor_name}: Wrong number of dimensions, expected {len(expected_shape)}, got {len(tensor.shape)}")
            return False
        
        for i, (expected, actual) in enumerate(zip(expected_shape, tensor.shape)):
            if expected is not None and actual != expected:
                print(f"❌ {tensor_name}: Dimension {i} mismatch, expected {expected}, got {actual}")
                return False
        
        # Check for NaN or infinite values
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            print(f"❌ {tensor_name}: Contains NaN or infinite values")
            return False
        
        return True
    
    def _validate_video_data(self, video, episode_name):
        """Validate video data and return True if valid."""
        if video is None:
            print(f"❌ Episode {episode_name}: Missing video data")
            return False
        
        if not isinstance(video, list) or len(video) == 0:
            print(f"❌ Episode {episode_name}: Invalid video format")
            return False
        
        # Check if all frames are valid
        for i, frame in enumerate(video):
            if frame is None:
                print(f"❌ Episode {episode_name}: Frame {i} is None")
                return False
            if not hasattr(frame, 'size'):
                print(f"❌ Episode {episode_name}: Frame {i} is not a PIL Image")
                return False
        
        return True
    
    def _validate_hand_motion(self, hand_motion, episode_name):
        """Validate hand motion data and return True if valid."""
        if hand_motion is None:
            return True  # Hand motion is optional
        
        # Check if it's a tensor
        if not isinstance(hand_motion, torch.Tensor):
            print(f"❌ Episode {episode_name}: Hand motion is not a tensor")
            return False
        
        # Check dimensions: should be [seq_len, feature_dim] or [batch, seq_len, feature_dim]
        if len(hand_motion.shape) not in [2, 3]:
            print(f"❌ Episode {episode_name}: Hand motion has wrong dimensions: {hand_motion.shape}")
            return False
        
        # Check feature dimension (should be 10 for single hand or 20 for dual hand)
        feature_dim = hand_motion.shape[-1]
        if feature_dim not in [10, 20]:
            print(f"❌ Episode {episode_name}: Hand motion has wrong feature dimension: {feature_dim}")
            return False
        
        # Check for NaN or infinite values
        if torch.isnan(hand_motion).any() or torch.isinf(hand_motion).any():
            print(f"❌ Episode {episode_name}: Hand motion contains NaN or infinite values")
            return False
        
        return True
    
    def _validate_object_trajectory(self, trajectory, object_ids, episode_name):
        """Validate object trajectory data and return True if valid."""
        if trajectory is None and object_ids is None:
            return True  # Both are optional
        
        # If one is provided, both should be provided
        if (trajectory is None) != (object_ids is None):
            print(f"❌ Episode {episode_name}: Inconsistent object data - trajectory: {trajectory is not None}, ids: {object_ids is not None}")
            return False
        
        if trajectory is not None:
            if not isinstance(trajectory, torch.Tensor):
                print(f"❌ Episode {episode_name}: Object trajectory is not a tensor")
                return False
            
            # Check dimensions: should be [seq_len, num_objects, 9] or [batch, seq_len, num_objects, 9]
            if len(trajectory.shape) not in [3, 4]:
                print(f"❌ Episode {episode_name}: Object trajectory has wrong dimensions: {trajectory.shape}")
                return False
            
            # Check feature dimension (should be 9 for 3D position + 6D rotation)
            feature_dim = trajectory.shape[-1]
            if feature_dim != 9:
                print(f"❌ Episode {episode_name}: Object trajectory has wrong feature dimension: {feature_dim}")
                return False
            
            # Check for NaN or infinite values
            if torch.isnan(trajectory).any() or torch.isinf(trajectory).any():
                print(f"❌ Episode {episode_name}: Object trajectory contains NaN or infinite values")
                return False
        
        if object_ids is not None:
            if not isinstance(object_ids, torch.Tensor):
                print(f"❌ Episode {episode_name}: Object IDs is not a tensor")
                return False
            
            # Check dimensions: should be [num_objects] or [batch, num_objects]
            if len(object_ids.shape) not in [1, 2]:
                print(f"❌ Episode {episode_name}: Object IDs has wrong dimensions: {object_ids.shape}")
                return False
        
        return True
    
    def _validate_embodiment_image(self, image, episode_name):
        """Validate embodiment image data and return True if valid."""
        if image is None:
            return True  # Embodiment image is optional
        
        if not hasattr(image, 'size'):
            print(f"❌ Episode {episode_name}: Embodiment image is not a PIL Image")
            return False
        
        return True
    
    def _validate_episode_data(self, data, episode_name):
        """Comprehensive validation of episode data."""
        is_valid = True
        
        # Validate video data (required)
        if not self._validate_video_data(data.get('video'), episode_name):
            is_valid = False
        
        # Validate hand motion (optional)
        if not self._validate_hand_motion(data.get('hand_motion_sequence'), episode_name):
            is_valid = False
        
        # Validate object trajectory (optional)
        if not self._validate_object_trajectory(
            data.get('object_trajectory_sequence'), 
            data.get('object_ids'), 
            episode_name
        ):
            is_valid = False
        
        # Validate embodiment image (optional)
        if not self._validate_embodiment_image(data.get('embodiment_image'), episode_name):
            is_valid = False
        
        # Validate prompt
        if not data.get('prompt') or not isinstance(data['prompt'], str):
            print(f"❌ Episode {episode_name}: Missing or invalid prompt")
            is_valid = False
        
        return is_valid
    
    def __getitem__(self, data_id):
        """Load episode data with all modalities and validate."""
        episode = self.episodes[data_id % len(self.episodes)]
        episode_path = episode['episode_path']
        episode_name = episode['episode_name']
        task_name = episode['task_name']
        task_prompt = self.generate_task_prompt(task_name)
        
        data = {
            'task_name': task_name,
            'episode_name': episode_name,
            'prompt': task_prompt,
            # Add CLIP labels for contrastive learning
            'task_label': self.get_task_label(task_prompt),
            'embodiment_label': self.get_embodiment_label(task_name),
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
        
        # Comprehensive validation of episode data
        if not self._validate_episode_data(data, episode_name):
            print(f"❌ Episode {episode_name}: Failed validation, skipping")
            return None
        
        print(f"✅ Episode {episode_name}: Passed validation")
        return data
    
    def __len__(self):
        """Dataset length with repeat factor."""
        return len(self.episodes) * self.repeat


def create_training_dataset(args):
    """Factory function to create VideoDatasetE for training."""
    return VideoDatasetE(args=args) 


# test load_object_trajectory_from_hdf5
if __name__ == "__main__":
    from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, launch_training_task, wan_parser, enable_club_training_defaults
    # test load_object_trajectory_from_hdf5
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
    
    # Enable appropriate defaults for CLUB training
    args = enable_club_training_defaults(args)
    
    # Parse VACE-E layers
    vace_e_layers = tuple(map(int, args.vace_e_layers.split(","))) if args.vace_e_layers else (0, 5, 10, 15, 20, 25)
    
    # Create dataset with VACE-E support
    dataset = create_training_dataset(args)
    data = dataset[0]
    print(data)
    obj_traj, obj_ids = dataset.load_object_trajectory_from_hdf5("/home/zhiyuan/Code/small_dataset/104-lars-grasping_2024-11-08_15-23-40/episode_23/episode_23_object_trajectories.hdf5")
    print(obj_traj.shape)
    print(obj_ids)