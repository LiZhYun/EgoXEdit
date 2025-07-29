# VACE-E Training Pipeline

Enhanced training pipeline for **VACE-E** (Video Animation and Control Engine - Enhanced) with robot demonstration support, dual-hand motion control, and task-embodiment fusion.

## 🎯 Overview

The VACE-E training pipeline extends the standard WAN video generation with:
- **Dual-hand motion sequences** (20D: left wrist + right wrist + grippers)  
- **Object trajectory tracking** (multi-object 9D pose sequences)
- **Task-specific prompt generation** (dynamic based on metadata)
- **Embodiment image integration** (end-effector camera views)
- **Multi-modal task-embodiment fusion** (weighted attention mechanisms)

## 📁 File Structure

```
examples/wanvideo/model_training/
├── train_E.py              # Enhanced training script with VACE-E support
├── dataset_E.py            # Extended dataset with HDF5 robot data loading
├── configs/                # Pre-configured training scenarios
│   ├── vace_e_full.yaml   # Full model training (high-end GPUs)
│   ├── vace_e_lora.yaml   # LoRA efficient training (mid-range GPUs)  
│   └── vace_e_dev.yaml    # Development/debugging (any GPU)
└── README.md              # This documentation
```

## 🚀 Quick Start

### **Prerequisites**

Before training, ensure you have the required models:

```bash
# Core VACE-E models (automatically downloaded)
- Wan-AI/Wan2.1-VACE-1.3B (DiT, T5, VAE)

# Required for embodiment image encoding
- Wan-AI/Wan2.1-I2V-14B-480P (CLIP model)
```

**Note**: The CLIP model is essential for encoding end-effector images. It will be automatically downloaded during training.

### 1. **Development Testing** (Fastest setup)
```bash
# Quick test with minimal resources
python train_E.py \
    --config configs/vace_e_dev.yaml \
    --dataset_base_path "/home/zhiyuan/Codes/DataSets/small_test" \
    --task_metadata_path "/home/zhiyuan/Codes/human-policy/data/ph2d_metadata.json" \
    --output_path "./debug_checkpoints"
```

### 2. **LoRA Training** (Recommended for most users)
```bash
# Memory-efficient training with LoRA
python train_E.py \
    --config configs/vace_e_lora.yaml \
    --dataset_base_path "/home/zhiyuan/Codes/DataSets/small_test" \
    --task_metadata_path "/home/zhiyuan/Codes/human-policy/data/ph2d_metadata.json" \
    --output_path "./lora_checkpoints"
```

### 3. **Full Model Training** (Research/Production)
```bash
# Complete model training (requires high-end GPU)
python train_E.py \
    --config configs/vace_e_full.yaml \
    --dataset_base_path "/home/zhiyuan/Codes/DataSets/small_test" \
    --task_metadata_path "/home/zhiyuan/Codes/human-policy/data/ph2d_metadata.json" \
    --output_path "./full_model_checkpoints"
```

## ⚙️ Configuration Options

### 📋 **Training Configurations**

| Configuration | Use Case | Hardware Requirements | Training Time |
|---|---|---|---|
| **`vace_e_dev.yaml`** | Testing, debugging, CI/CD | GTX 1080+ (8GB VRAM) | 5-15 minutes |
| **`vace_e_lora.yaml`** | Fine-tuning, adaptation | RTX 3080+ (12GB VRAM) | 12-24 hours |
| **`vace_e_full.yaml`** | Research, production models | RTX 4090/A100+ (24GB VRAM) | 3-5 days |

### 🔧 **Key Parameters**

#### VACE-E Model Configuration
```yaml
enable_vace_e: true                    # Enable/disable VACE-E features
vace_e_layers: "0,5,10,15,20,25"      # DiT layer indices for VACE-E integration
vace_e_task_processing: true           # Enable task feature processing
```

#### Robot Data Configuration  
```yaml
robot_data_path: "/path/to/robot/data"              # HDF5 demonstration data
task_metadata_path: "/path/to/metadata.json"       # Task descriptions and metadata
max_hand_motion_length: 512                        # Maximum motion sequence length
max_object_trajectory_length: 512                  # Maximum trajectory length
max_objects: 10                                    # Maximum tracked objects
fallback_to_video_only: true                      # Allow training without robot data
```

#### Training Strategy
```yaml
training_stages:
  stage_1:
    description: "Task encoder warm-up"
    epochs: 10
    trainable_models: "vace_e"
    learning_rate: 5e-5
```

## 📊 Data Format

### Expected Directory Structure
```
/home/zhiyuan/Codes/DataSets/small_test/           # Base dataset path
├── task_folder_1_2024-11-08_15-23-40/            # Task folder with timestamp
│   ├── episode_0/
│   │   ├── episode_0.mp4                         # Target video (what we want to generate)
│   │   ├── episode_0_hand_trajectories.hdf5      # Dual-hand motion data
│   │   ├── episode_0_object_trajectories.hdf5    # Object trajectory data
│   │   ├── episode_0_hands_masked.mp4            # VACE control video (hands masked)
│   │   ├── episode_0_hands_mask.mp4              # VACE mask video
│   │   └── end_effector_camera.jpg               # Embodiment image
│   └── episode_1/...
└── task_folder_2_2024-11-13_14-40-07/
    └── ...

# Separate metadata file
/home/zhiyuan/Codes/human-policy/data/ph2d_metadata.json  # Task metadata
```

### Video File Types

The dataset distinguishes between three types of video files:

1. **Target Video** (`episode_N.mp4`): 
   - The **ground truth video** we want the model to generate
   - Clean, original episode recording
   - Used as training target

2. **VACE Control Video** (`episode_N_hands_masked.mp4`):
   - Hands-masked version for VACE control
   - Provides spatial guidance during generation
   - Shows context but masks hand regions

3. **VACE Mask Video** (`episode_N_hands_mask.mp4`):
   - Binary mask indicating hand regions
   - Defines which areas to focus VACE control on
   - Used for attention masking

### Task Metadata Format

The training pipeline uses **ph2d_metadata.json** for task information and dynamic prompt generation:

```json
{
    "metadata": {
        "task_types": ["pouring", "grasping", "picking", ...],
        "embodiment_type_camera": {...},
        "cameras": {...}
    },
    "per_task_attributes": {
        "104-lars-grasping_2024-11-08_15-23-40": {
            "task_type": "grasping",
            "embodiment_type": "human_avp", 
            "left_hand": false,
            "right_hand": true,
            "objects": "white bottle with black lid."
        },
        "902-pouring-val-2024_11_18-18_49_25": {
            "task_type": "pouring",
            "embodiment_type": "h1_inspire",
            "left_hand": true,
            "right_hand": true, 
            "objects": "cup, bottle."
        }
    }
}
```

**Generated prompts**: 
- `"Right hand grasping white bottle with black lid."`
- `"Pouring, left hand cup, right hand bottle"`

**Key points**:
- Task data is under `"per_task_attributes"`
- Task IDs are full folder names (match dataset folder structure)
- Additional metadata includes camera calibration and embodiment types

### HDF5 Data Format

#### Hand Trajectories (`*_hand_trajectories.hdf5`)
**Dual-hand format (20D output)**:
```python
{
    # Preferred dual-hand format
    'left_wrist': {
        'positions': [frames, 3],        # Left wrist position
        'rotations_6d': [frames, 6],     # Left wrist rotation (6D)
    },
    'right_wrist': {
        'positions': [frames, 3],        # Right wrist position  
        'rotations_6d': [frames, 6],     # Right wrist rotation (6D)
    },
    'left_hand_states': [frames],        # Left gripper states (binary)
    'right_hand_states': [frames],       # Right gripper states (binary)
    
    # Single-hand format (legacy support, converted to dual-hand)
    'right_wrist': {
        'positions': [frames, 3],        # Right wrist position → right hand
        'rotations_6d': [frames, 6],     # Right wrist rotation → right hand
    },
    'right_hand_states': [frames],       # Right gripper states → right hand
}
```

**Note**: Rotations are 6D (used as-is, no extension). Gripper states converted to binary (closed=0, open=1). Single-hand data converted to dual-hand with left hand as zeros.

**IMPORTANT**: All sequences use their **natural lengths** without padding! The VACE-E model handles variable-length sequences using attention masks, exactly like the inference pipeline.

#### Object Trajectories (`*_object_trajectories.hdf5`)
```python
{
    'object_0': {
        'positions': [frames, 3],        # Object 0 position
        'rotations_6d': [frames, 6],     # Object 0 rotation (6D)
        'attrs': {'object_id': 0},       # Object identifier
    },
    'object_1': {
        'positions': [frames, 3],        # Object 1 position
        'rotations_6d': [frames, 6],     # Object 1 rotation (6D)
        'attrs': {'object_id': 1},       # Object identifier
    },
    # ... additional objects
}
```

### Dataset Discovery Process

The dataset automatically discovers episodes by:

1. **Scanning task folders** in `/home/zhiyuan/Codes/DataSets/small_test/`
2. **Using full folder names** as task identifiers (matches ph2d_metadata.json keys)
   - `104-lars-grasping_2024-11-08_15-23-40` → exactly as stored in metadata
3. **Finding episode folders** within each task (`episode_0`, `episode_1`, etc.)
4. **Validating required files** (hands_masked.mp4, hands_mask.mp4)
5. **Loading optional data** (HDF5 files, embodiment images)
6. **Generating prompts** from ph2d_metadata.json["per_task_attributes"] using full task names

### Data Loading Pipeline

For each episode, the dataset loads:

- **Target video**: `episode_N.mp4` (training ground truth)
- **VACE control video**: `episode_N_hands_masked.mp4` (spatial guidance)
- **VACE mask**: `episode_N_hands_mask.mp4` (attention masking)
- **Hand motion**: `episode_N_hand_trajectories.hdf5` → 20D tensor
- **Object trajectories**: `episode_N_object_trajectories.hdf5` → [seq_len, num_objects, 9]
- **Embodiment image**: `end_effector_*.jpg` (VACE reference + embodiment)
- **Task prompt**: Generated from ph2d_metadata.json

### Training Data Flow

```
Target Video (episode_N.mp4)           → Training Ground Truth
    ↓
VACE Control (episode_N_hands_masked)   → Spatial Guidance  
    ↓                                   
VACE Mask (episode_N_hands_mask)       → Attention Masking
    ↓
Hand Motion (HDF5) + Object Trajectories → VACE-E Task Features
    ↓
Embodiment Image + Task Prompt          → VACE-E Embodiment Features
    ↓
Model learns to generate Target Video given all control inputs
```

## 🎛️ Advanced Usage

### Custom Configuration Override
```bash
# Override config parameters from command line
python train_E.py \
    --config configs/vace_e_lora.yaml \
    --learning_rate 1e-4 \
    --vace_e_layers "0,3,6,9,12,15,18,21,24" \
    --batch_size 4
```

### Progressive Training
```yaml
# In your config file
training_stages:
  stage_1:
    description: "Text and motion encoder warm-up"
    epochs: 15
    trainable_models: "vace_e"
    learning_rate: 5e-4
    
  stage_2:
    description: "Joint VACE-E and DiT training"  
    epochs: 20
    trainable_models: "dit,vace_e"
    learning_rate: 1e-5
```

### Memory Optimization
```yaml
# For limited VRAM
vram_optimization:
  enabled: true
  offload_optimizer: true      # Move optimizer to CPU
  cpu_offload: true           # Offload frozen models
  gradient_compression: 0.5    # Compress gradients
  
use_gradient_checkpointing: true
use_gradient_checkpointing_offload: true
```

## 🔍 Monitoring and Debugging

### Logging Integration
```yaml
logging:
  wandb_project: "vace_e_training"
  experiment_name: "my_experiment"
  log_frequency: 100
  save_model_artifacts: true
```

### Debug Mode
```bash
# Enable comprehensive debugging
python train_E.py \
    --config configs/vace_e_dev.yaml \
    --debug True \
    --verbose_logging True \
    --save_intermediate_outputs True
```

### Single Sample Overfitting Test
```bash
# Test if model can overfit single sample (debugging)
python train_E.py \
    --config configs/vace_e_dev.yaml \
    --dataset_repeat 100 \
    --num_epochs 10 \
    --batch_size 1
```

## 📈 Performance Benchmarks

### Training Speed (approximate)

| Configuration | GPU | Batch Size | Samples/sec | VRAM Usage |
|---|---|---|---|---|
| Dev | RTX 2060 | 1 | 0.5 | 8GB |
| LoRA | RTX 3080 | 2 | 0.8 | 12GB |  
| LoRA | RTX 4090 | 4 | 1.5 | 16GB |
| Full | A100 | 2 | 0.3 | 32GB |

### Memory Usage Breakdown
- **Base Model**: ~8GB (DiT + VAE + Text Encoder)
- **VACE-E Components**: ~2GB (Task encoders + fusion)
- **Training Overhead**: ~4-8GB (gradients + optimizer)
- **Video Processing**: ~2-4GB (frame buffers)

## 🚨 Troubleshooting

### Common Issues

#### 1. **CUDA Out of Memory**
```bash
# Solutions (in order of preference):
# 1. Use gradient checkpointing
--use_gradient_checkpointing_offload True

# 2. Reduce batch size or resolution
--batch_size 1 --height 240 --width 416

# 3. Enable CPU offloading  
--vram_optimization True --cpu_offload True

# 4. Use development config
--config configs/vace_e_dev.yaml
```

#### 2. **Data Loading Errors**
```bash
# Check HDF5 file structure
python -c "import h5py; print(list(h5py.File('trajectory.hdf5', 'r').keys()))"

# Enable fallback mode
--fallback_to_video_only True
```

#### 3. **Device Mismatch Errors**
```bash
# Usually resolved by:
# 1. Ensuring consistent tensor devices in VACE-E components
# 2. Using latest version with device management fixes
# 3. Restarting training with clean environment
```

#### 4. **Slow Training**
```bash
# Optimization strategies:
# 1. Use LoRA instead of full training
--config configs/vace_e_lora.yaml

# 2. Reduce sequence lengths
--max_hand_motion_length 256 --max_object_trajectory_length 256

# 3. Use mixed precision
--mixed_precision bf16

# 4. Enable model compilation (PyTorch 2.0+)
--compile_model True
```

## 🧪 Validation and Testing

### Built-in Validation
```yaml
validation:
  enabled: true
  frequency: 5              # Every 5 epochs
  num_samples: 10
  save_videos: true
  metrics:
    - "video_quality"
    - "task_alignment" 
    - "motion_accuracy"
```

### Custom Validation Scripts
```bash
# Run validation on specific checkpoint
python validate_E.py \
    --checkpoint "/path/to/checkpoint.pt" \
    --test_data "/path/to/test/data" \
    --output_dir "./validation_results"
```

## 📚 Integration Examples

### With Existing Inference Pipeline
```python
# Load trained VACE-E model
pipe = WanVideoPipeline.from_pretrained(
    model_configs=[...],
    enable_vace_e=True,
    vace_e_checkpoint="/path/to/trained/vace_e.pt"
)

# Use with robot demonstration data
video = pipe(
    prompt="Right hand grasping red bottle", 
    vace_e_hand_motion_sequence=hand_motion_data,
    vace_e_object_trajectory_sequence=object_data,
    vace_e_embodiment_image_features=camera_features
)
```

### Custom Loss Functions
```python
class CustomVACELoss(torch.nn.Module):
    def forward(self, predicted, target, vace_e_context):
        # Standard video loss
        video_loss = F.mse_loss(predicted, target)
        
        # VACE-E guidance loss
        vace_e_loss = self.compute_task_alignment_loss(predicted, vace_e_context)
        
        return video_loss + 0.5 * vace_e_loss
```

## 🔮 Future Extensions

### Planned Features
- [ ] **Multi-modal LoRA**: Task-specific adapter switching
- [ ] **Curriculum Learning**: Progressive complexity increase  
- [ ] **Self-supervised Pre-training**: Learn from unlabeled robot data
- [ ] **Distributed Training**: Multi-GPU and multi-node support
- [ ] **Quantization Support**: INT8/INT4 training for efficiency
- [ ] **Online Learning**: Continuous adaptation during deployment

### Research Directions
- [ ] **Hierarchical Task Decomposition**: Multi-level task understanding
- [ ] **Cross-embodiment Transfer**: Human → Robot adaptation
- [ ] **Temporal Attention**: Advanced temporal modeling
- [ ] **Physics-aware Training**: Incorporate physical constraints

## 📖 References

- [Original VACE Paper](link-to-paper)
- [Diffusion Training Best Practices](link-to-guide)
- [Robot Demonstration Collection](link-to-dataset)
- [LoRA Fine-tuning Guide](link-to-lora-guide)

## 💬 Support

For issues and questions:
1. Check this README and troubleshooting section
2. Review configuration files for examples
3. Test with development config first
4. Open GitHub issue with reproducible example

---

**Happy Training!** 🤖🎬✨ 