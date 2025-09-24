"""
Representation Extraction and t-SNE Visualization for VACE-E

This script extracts task and embodiment representations from the trained VACE-E model
without generating full videos, then creates t-SNE visualizations to analyze the
learned representations similar to the bias analysis shown in the reference image.

Key Features:
1. Extract task and embodiment features from different episodes/tasks
2. Save representations to disk for analysis
3. Generate t-SNE plots with different bias conditions
4. Support for multiple tasks and embodiments for comparison
5. Lightweight inference (no video generation)

The script extracts representations at the point where they would be used for
task-embodiment fusion in the VACE-E model, providing insights into how well
the model has learned to disentangle task and embodiment information.
"""

import torch
import numpy as np
import h5py
import json
import argparse
import os
import glob
from PIL import Image
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pickle
from tqdm import tqdm

from diffsynth import load_state_dict
from diffsynth.pipelines.wan_video_new_E import WanVideoPipeline, ModelConfig
import tempfile
import shutil


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Extract representations and create t-SNE visualization")
    
    # Model and weights
    parser.add_argument("--state_dict_path", type=str, 
                       default="/scratch/work/liz23/h2rego_video_generation/models/train/Wan2.1-VACE-E-1.3B_full_no_club/epoch-2.safetensors",
                       help="Path to the state dict file to load")
    
    # Data paths
    parser.add_argument("--dataset_path", type=str, 
                       default="/scratch/work/liz23/DataSets/picking",
                       help="Path to the dataset folder containing robot demonstration data")
    
    parser.add_argument("--end_effector_image", type=str,
                       default="/scratch/work/liz23/DataSets/PH2D_videos/405-pick_on_color_pad_right_far_far-2025_01_13-19_29_04/episode_7/episode_7_hands_only.jpg",
                       help="Path to the end-effector image to use")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="representations_analysis",
                       help="Output directory for saved representations and plots")
    
    parser.add_argument("--max_episodes", type=int, default=200,
                       help="Maximum number of episodes to process for representation extraction")
    
    parser.add_argument("--tsne_perplexity", type=float, default=30.0,
                       help="Perplexity parameter for t-SNE")
    
    parser.add_argument("--tsne_components", type=int, default=2,
                       help="Number of components for t-SNE")
    
    return parser.parse_args()


def load_task_metadata(metadata_path="/scratch/work/liz23/h2rego_data_preprocess/data/ph2d_metadata.json"):
    """Load task metadata from JSON file."""
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"✅ Loaded metadata from: {metadata_path}")
        return metadata
    except Exception as e:
        print(f"⚠️ Failed to load metadata from {metadata_path}: {e}")
        return None


def generate_task_prompt(task_name, metadata):
    """Generate robot task prompt based on metadata."""
    task_attrs = metadata["per_task_attributes"].get(task_name, {})
    task_type = task_attrs.get("task_type", "manipulation")
    objects = task_attrs.get("objects", "objects")
    left_hand = task_attrs.get("left_hand", False)
    right_hand = task_attrs.get("right_hand", True)

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


def load_robot_data_from_hdf5(dataset_path):
    """Load robot demonstration data from HDF5 files, similar to the original script."""
    print(f"Loading robot data from: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset path does not exist: {dataset_path}")
        return []
    
    task_folders = [f for f in os.listdir(dataset_path) 
                   if os.path.isdir(os.path.join(dataset_path, f)) and not f.startswith('.')]
    
    if not task_folders:
        print(f"❌ No task folders found in: {dataset_path}")
        return []
    
    print(f"Found {len(task_folders)} task folders: {task_folders}")
    
    all_episodes = []
    
    for task_folder in task_folders:
        task_path = os.path.join(dataset_path, task_folder)
        print(f"\n📁 Processing task: {task_folder}")
        
        episode_folders = [f for f in os.listdir(task_path) 
                          if os.path.isdir(os.path.join(task_path, f)) and f.startswith('episode_')]
        
        if not episode_folders:
            print(f"   ⚠️ No episode folders found in {task_folder}")
            continue
        
        print(f"   Found {len(episode_folders)} episodes: {episode_folders}")
        
        for episode_folder in episode_folders:
            episode_path = os.path.join(task_path, episode_folder)
            episode_name = episode_folder
            
            print(f"   📂 Processing episode: {episode_name}")
            
            hand_hdf5_path = os.path.join(episode_path, f"{episode_name}_hand_trajectories.hdf5")
            
            episode_data = {
                'task_name': task_folder,
                'episode_name': episode_name,
                'episode_path': episode_path,
                'hand_motion_sequence': None,
                'object_trajectory_sequence': None,
                'object_ids': None,
                'end_effector_image_path': None
            }
            
            # Load hand motion data
            if os.path.exists(hand_hdf5_path):
                try:
                    print(f"      Loading hand trajectories from: {hand_hdf5_path}")
                    with h5py.File(hand_hdf5_path, 'r') as f:
                        if 'left_wrist' in f and 'right_wrist' in f:
                            # Load dual-hand data
                            left_positions = f['left_wrist/positions'][:]
                            left_rotations = f['left_wrist/rotations_6d'][:]
                            right_positions = f['right_wrist/positions'][:]
                            right_rotations = f['right_wrist/rotations_6d'][:]
                            
                            # Create 20D dual-hand motion sequence
                            left_gripper_states = np.zeros((len(left_positions), 1))
                            right_gripper_states = np.zeros((len(right_positions), 1))
                            
                            dual_hand_motion = np.concatenate([
                                left_positions, left_rotations, left_gripper_states,
                                right_positions, right_rotations, right_gripper_states
                            ], axis=1)
                            
                            episode_data['hand_motion_sequence'] = dual_hand_motion
                            print(f"      ✓ Dual-hand motion: {dual_hand_motion.shape}")
                
                except Exception as e:
                    print(f"      ❌ Error loading hand trajectories: {e}")
            
            # Look for end-effector images
            image_pattern = os.path.join(episode_path, f"{episode_name}_hands_only.*")
            image_files = glob.glob(image_pattern)
            if image_files:
                episode_data['end_effector_image_path'] = image_files[0]
                print(f"      ✓ End-effector image: {os.path.basename(episode_data['end_effector_image_path'])}")
            
            all_episodes.append(episode_data)
    
    print(f"\n✅ Loaded {len(all_episodes)} episodes from {len(task_folders)} tasks")
    return all_episodes


class RepresentationExtractor:
    """Class to extract representations from the VACE-E model."""
    
    def __init__(self, model_path, device="cuda"):
        """Initialize the representation extractor."""
        self.device = device
        self.model_path = model_path
        
        # Load WanVideoPipeline with VACE-E support
        print("Loading WanVideoPipeline with VACE-E support...")
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=[
                ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
                ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
                ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
                ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-480P", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", offload_device="cpu")
            ],
            enable_vace_e=True,
            vace_e_layers=(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28),
            vace_e_task_processing=True,
        )
        
        # Load trained weights
        state_dict = load_state_dict(model_path)
        self.pipe.vace_e.load_state_dict(state_dict)
        self.pipe.enable_vram_management()
        
        print("✅ Model loaded successfully!")
    
    def extract_representations(self, episode_data, task_prompt, end_effector_image_path):
        """
        Extract task and embodiment representations for a single episode.
        
        Returns:
            tuple: (task_features, embodiment_features) or (None, None) if extraction fails
        """
        try:
            # Prepare minimal inputs needed for the model
            prompt = task_prompt if task_prompt else "Robot manipulation task"
            
            # Prepare VACE-E context
            vace_e_context = {}
            
            # Process text features using prompter (same as main script)
            text_features = self.pipe.prompter.encode_prompt(prompt, device=self.device)
            vace_e_context['text_features'] = text_features
            vace_e_context['text_mask'] = torch.ones(1, text_features.shape[1], device=self.device)
            
            # Process hand motion sequence
            if episode_data['hand_motion_sequence'] is not None:
                motion_tensor = torch.from_numpy(episode_data['hand_motion_sequence']).float().unsqueeze(0).to(self.device)
                vace_e_context['hand_motion_sequence'] = motion_tensor
                vace_e_context['motion_mask'] = torch.ones(motion_tensor.shape[:2], device=self.device)
            
            # Process end-effector image
            if os.path.exists(episode_data['end_effector_image_path']):
                end_effector_image = Image.open(episode_data['end_effector_image_path']).resize((832, 480))
                end_effector_image = self.pipe.preprocess_image(end_effector_image.resize((832, 480))).to(self.pipe.device)
                embodiment_features = self.pipe.image_encoder.encode_image([end_effector_image])
                vace_e_context['embodiment_image_features'] = embodiment_features
            
            # Create a minimal latent tensor for the forward pass
            # This is needed to trigger the VACE-E processing
            batch_size = 1
            height, width, num_frames = 480, 832, 81
            latent_height = height // 8
            latent_width = width // 8
            latent_frames = (num_frames - 1) // 4 + 1
            
            x = torch.randn(
                batch_size, 16, latent_frames, latent_height, latent_width,
                device=self.device, dtype=self.pipe.torch_dtype
            )
            
            # Prepare timestep
            timestep = torch.tensor([500], device=self.device)
            
            # Clear any existing global features
            if '_current_task_features' in globals():
                del globals()['_current_task_features']
            if '_current_embodiment_features' in globals():
                del globals()['_current_embodiment_features']
            
            # Call the VACE-E model with return_intermediate_features=True
            with torch.no_grad():
                if hasattr(self.pipe, 'vace_e') and self.pipe.vace_e is not None:
                    # Prepare required arguments for VACE-E
                    dit = self.pipe.dit
                    
                    # Process timestep to get t_mod (fix dtype issue)
                    from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d
                    # Convert timestep to float to match model dtype
                    timestep_float = timestep.float().to(self.pipe.torch_dtype)
                    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep_float))
                    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
                    
                    # Process text to get context
                    context = dit.text_embedding(vace_e_context.get('text_features'))
                    
                    # Patchify the input tensor (required before VACE-E processing)
                    x_patchified, (f, h, w) = dit.patchify(x, None)  # No camera control for representation extraction
                    
                    # Create frequency embeddings using patchified dimensions
                    freqs = torch.cat([
                        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
                    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
                    
                    # Call the VACE-E forward pass with intermediate features
                    vace_e_result = self.pipe.vace_e(
                        x_patchified, None, context, t_mod, freqs,  # Use patchified input
                        # Task features
                        text_features=vace_e_context.get('text_features'),
                        hand_motion_sequence=vace_e_context.get('hand_motion_sequence'),
                        object_trajectory_sequence=vace_e_context.get('object_trajectory_sequence'),
                        object_ids=vace_e_context.get('object_ids'),
                        text_mask=vace_e_context.get('text_mask'),
                        motion_mask=vace_e_context.get('motion_mask'),
                        trajectory_mask=vace_e_context.get('trajectory_mask'),
                        # Embodiment features
                        embodiment_image_features=vace_e_context.get('embodiment_image_features'),
                        # Enable intermediate feature extraction
                        return_intermediate_features=True
                    )
                    
                    # Handle the return value - VACE-E returns (vace_e_hints, task_features, embodiment_features) when return_intermediate_features=True
                    if isinstance(vace_e_result, tuple) and len(vace_e_result) == 3:
                        _, task_features_direct, embodiment_features_direct = vace_e_result
                        # Use direct returns if available
                        task_features = task_features_direct
                        embodiment_features = embodiment_features_direct
                    else:
                        # Fallback to globals if direct return not available
                        task_features = globals().get('_current_task_features', None)
                        embodiment_features = globals().get('_current_embodiment_features', None)
                    
                    # Convert to numpy and detach from GPU (fix BFloat16 issue)
                    if task_features is not None:
                        task_features = task_features.detach().cpu().float().numpy()
                    if embodiment_features is not None:
                        embodiment_features = embodiment_features.detach().cpu().float().numpy()
                    
                    # Clean up global variables
                    if '_current_task_features' in globals():
                        del globals()['_current_task_features']
                    if '_current_embodiment_features' in globals():
                        del globals()['_current_embodiment_features']
                    
                    return task_features, embodiment_features
                
        except Exception as e:
            print(f"❌ Error extracting representations: {e}")
            import traceback
            traceback.print_exc()
            return None, None
        
        return None, None


def create_tsne_visualization(representations_data, output_dir):
    """Create t-SNE visualization of the extracted representations."""
    
    # Prepare data for t-SNE
    task_features_list = []
    embodiment_features_list = []
    task_labels = []
    embodiment_labels = []
    episode_info = []
    
    for episode_data, task_repr, embodiment_repr in representations_data:
        if task_repr is not None and embodiment_repr is not None:
            # Flatten features if they have multiple dimensions
            if task_repr.ndim > 2:
                task_repr = task_repr.reshape(task_repr.shape[0], -1)
            if embodiment_repr.ndim > 2:
                embodiment_repr = embodiment_repr.reshape(embodiment_repr.shape[0], -1)
            
            # Take mean across sequence dimension if present
            if task_repr.ndim == 2:
                task_repr = task_repr.mean(axis=0)
            if embodiment_repr.ndim == 2:
                embodiment_repr = embodiment_repr.mean(axis=0)
            
            task_features_list.append(task_repr)
            embodiment_features_list.append(embodiment_repr)
            task_labels.append(episode_data['task_name'])
            embodiment_labels.append(episode_data['episode_name'])
            episode_info.append(f"{episode_data['task_name']}_{episode_data['episode_name']}")
    
    if len(task_features_list) == 0:
        print("❌ No valid representations found for t-SNE visualization")
        return
    
    # Convert to numpy arrays
    task_features = np.array(task_features_list)
    embodiment_features = np.array(embodiment_features_list)
    
    print(f"Task features shape: {task_features.shape}")
    print(f"Embodiment features shape: {embodiment_features.shape}")
    
    # Create t-SNE plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('t-SNE Visualization of Task and Embodiment Representations', fontsize=16)
    
    # 1. Task features colored by task type
    if len(np.unique(task_labels)) > 1:
        tsne_task = TSNE(n_components=2, perplexity=min(30, len(task_features)//4), random_state=42)
        task_tsne = tsne_task.fit_transform(task_features)
        
        # Create color mapping for consistent colors
        unique_tasks = list(set(task_labels))
        color_map = plt.cm.tab20(np.linspace(0, 1, len(unique_tasks)))
        
        # Plot each task type with its own color
        for i, task in enumerate(unique_tasks):
            mask = np.array(task_labels) == task
            axes[0, 0].scatter(task_tsne[mask, 0], task_tsne[mask, 1], 
                             c=[color_map[i]], alpha=0.7, s=50, label=task[:20])
        
        axes[0, 0].set_title('Task Features (colored by task type)')
        axes[0, 0].set_xlabel('t-SNE 1')
        axes[0, 0].set_ylabel('t-SNE 2')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # 2. Embodiment features colored by episode
    if len(np.unique(embodiment_labels)) > 1:
        tsne_embodiment = TSNE(n_components=2, perplexity=min(30, len(embodiment_features)//4), random_state=42)
        embodiment_tsne = tsne_embodiment.fit_transform(embodiment_features)
        
        axes[0, 1].scatter(embodiment_tsne[:, 0], embodiment_tsne[:, 1], 
                          c=[hash(label) % 20 for label in embodiment_labels], 
                          cmap='tab20', alpha=0.7, s=50)
        axes[0, 1].set_title('Embodiment Features (colored by episode)')
        axes[0, 1].set_xlabel('t-SNE 1')
        axes[0, 1].set_ylabel('t-SNE 2')
    
    # 3. Combined features - task and embodiment together
    combined_features = np.concatenate([task_features, embodiment_features], axis=0)
    combined_labels = ['Task'] * len(task_features) + ['Embodiment'] * len(embodiment_features)
    
    tsne_combined = TSNE(n_components=2, perplexity=min(30, len(combined_features)//4), random_state=42)
    combined_tsne = tsne_combined.fit_transform(combined_features)
    
    task_mask = np.array(combined_labels) == 'Task'
    embodiment_mask = np.array(combined_labels) == 'Embodiment'
    
    axes[1, 0].scatter(combined_tsne[task_mask, 0], combined_tsne[task_mask, 1], 
                      c='blue', alpha=0.7, s=50, label='Task Features')
    axes[1, 0].scatter(combined_tsne[embodiment_mask, 0], combined_tsne[embodiment_mask, 1], 
                      c='red', alpha=0.7, s=50, label='Embodiment Features')
    axes[1, 0].set_title('Combined Task and Embodiment Features')
    axes[1, 0].set_xlabel('t-SNE 1')
    axes[1, 0].set_ylabel('t-SNE 2')
    axes[1, 0].legend()
    
    # 4. Bias analysis simulation (if we have enough diversity)
    # For now, we'll create a simple analysis based on task diversity
    if len(np.unique(task_labels)) >= 2:
        # Simulate different bias conditions by sampling subsets
        tsne_bias = TSNE(n_components=2, perplexity=min(20, len(task_features)//6), random_state=42)
        bias_tsne = tsne_bias.fit_transform(task_features)
        
        # Create different "bias conditions" based on task distribution
        unique_tasks = list(set(task_labels))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_tasks)))
        
        for i, task in enumerate(unique_tasks):
            mask = np.array(task_labels) == task
            axes[1, 1].scatter(bias_tsne[mask, 0], bias_tsne[mask, 1], 
                             c=[colors[i]], alpha=0.7, s=50, label=task[:15])
        
        axes[1, 1].set_title('Task Distribution Analysis')
        axes[1, 1].set_xlabel('t-SNE 1')
        axes[1, 1].set_ylabel('t-SNE 2')
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'tsne_representation_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ t-SNE plot saved to: {plot_path}")
    
    # Also save as PDF for publication quality
    plot_path_pdf = os.path.join(output_dir, 'tsne_representation_analysis.pdf')
    plt.savefig(plot_path_pdf, bbox_inches='tight')
    print(f"✅ t-SNE plot (PDF) saved to: {plot_path_pdf}")
    
    plt.show()


def main():
    """Main function to extract representations and create t-SNE visualization."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load robot demonstration data
    print("\n=== Loading Robot Demonstration Data ===")
    robot_episodes = load_robot_data_from_hdf5(args.dataset_path)
    
    # Load task metadata
    print("\n=== Loading Task Metadata ===")
    task_metadata = load_task_metadata("/scratch/work/liz23/h2rego_data_preprocess/data/ph2d_metadata.json")
    
    if not robot_episodes:
        print("❌ No robot episodes loaded, exiting...")
        return
    
    # Limit episodes for efficiency
    if len(robot_episodes) > args.max_episodes:
        robot_episodes = robot_episodes[:args.max_episodes]
        print(f"📊 Limited to {args.max_episodes} episodes for analysis")
    
    # Initialize representation extractor
    print("\n=== Initializing Representation Extractor ===")
    extractor = RepresentationExtractor(args.state_dict_path)
    
    # Extract representations for all episodes
    print("\n=== Extracting Representations ===")
    representations_data = []
    
    for episode_data in tqdm(robot_episodes, desc="Extracting representations"):
        # Generate task prompt
        task_prompt = generate_task_prompt(episode_data['task_name'], task_metadata)
        
        # Extract representations
        task_features, embodiment_features = extractor.extract_representations(
            episode_data, task_prompt, args.end_effector_image
        )
        
        if task_features is not None and embodiment_features is not None:
            representations_data.append((episode_data, task_features, embodiment_features))
            print(f"✅ Extracted representations for {episode_data['task_name']}_{episode_data['episode_name']}")
        else:
            print(f"⚠️ Failed to extract representations for {episode_data['task_name']}_{episode_data['episode_name']}")
    
    print(f"\n✅ Successfully extracted representations from {len(representations_data)} episodes")
    
    # Save representations to disk
    representations_file = os.path.join(args.output_dir, 'extracted_representations.pkl')
    with open(representations_file, 'wb') as f:
        pickle.dump(representations_data, f)
    print(f"💾 Representations saved to: {representations_file}")
    
    # Create t-SNE visualization
    print("\n=== Creating t-SNE Visualization ===")
    create_tsne_visualization(representations_data, args.output_dir)
    
    print(f"\n{'='*80}")
    print(f"🎉 REPRESENTATION ANALYSIS COMPLETE!")
    print(f"Processed {len(representations_data)} episodes")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
