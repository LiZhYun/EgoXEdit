# Egocentric Cross-Embodiment Video Editing via Dual Contrastive Representation Learning

**ICLR 2026 Submission**

![Framework](/assets/overview.png)

## Abstract

This repository contains the implementation of *Egocentric Cross-Embodiment Video Editing via Dual Contrastive Representation Learning*, a novel framework for egocentric cross-embodiment video editing by using a dual contrastive objective to disentangle a human demonstration and generate a coherent, robot-centric video.

## 📊 Method Overview

Our approach builds upon the [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) framework and introduces key innovations:

### 🔧 VACE-E Architecture
- **Task Processing**: Multi-modal fusion of text descriptions, dual-hand motions, and object trajectories
- **Embodiment Processing**: CLIP-encoded end-effector images for embodiment-specific context
- **Dual Contrastive Learning**: Separates task semantics from embodiment-specific features
- **Cross-Attention Fusion**: Aligns task and embodiment representations

### 🤖 Dual-Hand Motion Modeling
- **20D Motion Representation**: `[left_wrist(9), right_wrist(9), left_gripper(1), right_gripper(1)]`
- **Temporal Modeling**: CLS token-based sequence encoding for variable-length demonstrations
- **Multi-Object Tracking**: Simultaneous tracking of multiple objects with type embeddings

### 📈 Training Framework  
- **CLUB Loss**: Mutual information minimization between task and embodiment features
- **Contrastive Loss**: InfoNCE loss for representation learning
- **Flow Matching**: Continuous normalizing flows for video generation

## 🗂️ Dataset Structure

The **PH2D (Physical Human-Humanoid Data) Dataset** contains human and robot manipulation demonstrations:

```
data/PH2D_videos/
├── ph2d_metadata.json                    # Task metadata and camera parameters
├── robot_hands.jpg                       # Sample robot end-effector image
├── [task_name]/                          # Task-specific episodes
│   ├── episode_0.mp4                     # Target video (human/robot)
│   ├── episode_0_hands_masked.mp4 + episode_0_hands_mask.mp4      # Control video for VACE
│   ├── episode_0_hand_trajectories.hdf5  # Dual-hand motion data
│   ├── episode_0_object_trajectories.hdf5 # Object pose sequences
│   └── episode_0_hands_only.jpg          # End-effector image
```

### Task Categories
- **Manipulation Tasks**: `pouring`, `grasping`, `picking`
- **Embodiments**: `human`, `h1` 
- **Multi-Modal Data**: RGB videos, motion trajectories, object poses

## 🚀 Installation

### Prerequisites
- Python 3.10

### Setup Environment

```bash
# Clone the repository
git clone 
cd EgoXEdit

# Install dependencies
pip install -e .

# Install additional requirements
pip install -r requirements.txt
```

## 📋 Reproduction Instructions

### 1. Model Training

Train the model with dual contrastive learning:

```bash
./train.sh
```

### 2. Evaluation

Evaluate the trained model on cross-embodiment transfer tasks:

```bash
./eval.sh
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---