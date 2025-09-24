#!/bin/bash
#SBATCH --job-name=video_generation_tsne
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --time=0-01:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=./output/tsne.out
#SBATCH --error=./output/tsne.err

module load mamba
module load triton-dev/2025.1-gcc
module load gcc/13.3.0
module load cuda/12.6.2
export HF_HOME=/$WRKDIR/.huggingface_cache
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_EXTENSIONS_DIR=$WRKDIR/torch_extensions

source activate diffstudio

huggingface-cli login --token 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

# CUDA_VISIBLE_DEVICES=0 srun python ./save_object_trajectory.py # 
# CUDA_VISIBLE_DEVICES=0 srun python ./save_masked_obs.py # 10.5 hours with a100
# watch -n 1 nvidia-smi
# episode_data['end_effector_image_path'] = "/scratch/work/liz23/DataSets/PH2D_videos/405-pick_on_color_pad_right_far_far-2025_01_13-19_29_04/episode_7/episode_7_hands_only.jpg"
# episode_data['end_effector_image_path'] = "/scratch/work/liz23/DataSets/PH2D_videos/303-grasp_coke_random-2024_12_12-19_13_53/episode_5/episode_5_hands_only.jpg"
# episode_data['end_effector_image_path'] = "/scratch/work/liz23/DataSets/PH2D_videos/502-pouring_random-2025_01_10-20_21_26/episode_0/episode_0_hands_only.jpg"

srun python examples/wanvideo/model_inference/extract_representations_tsne.py \
    --state_dict_path /scratch/work/liz23/h2rego_video_generation/models/train/Wan2.1-VACE-E-1.3B_full/epoch-1.safetensors \
    --dataset_path /scratch/work/liz23/DataSets/PH2D_videos \
    --output_dir videos_club/1 \
    --end_effector_image /scratch/work/liz23/DataSets/robot_hands.jpg \