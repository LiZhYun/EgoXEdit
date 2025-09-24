#!/bin/bash
#SBATCH --job-name=video_generation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=1900G
#SBATCH --time=0-12:00:00
#SBATCH --gres=gpu:h200:8
#SBATCH --output=./output/output.out
#SBATCH --error=./output/error.err

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

srun accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml examples/wanvideo/model_training/train_E.py \
    --model_id_with_origin_paths "Wan-AI/Wan2.1-VACE-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-VACE-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-VACE-1.3B:Wan2.1_VAE.pth,Wan-AI/Wan2.1-I2V-14B-480P:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
    --enable_vace_e \
    --vace_e_layers "0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28" \
    --vace_e_task_processing \
    --task_metadata_path "/scratch/work/liz23/h2rego_data_preprocess/data/ph2d_metadata.json" \
    --dataset_base_path "/scratch/work/liz23/DataSets/PH2D_videos" \
    --output_path "./models/train/Wan2.1-VACE-E-1.3B_full" \
    --learning_rate "1e-4" \
    --num_epochs 1 \
    --num_frames 81 \
    --height 480 \
    --width 832 \
    --batch_size 5 \
    --num_workers 8 \
    --trainable_models "vace_e,club_estimator" \
    --remove_prefix_in_ckpt "pipe.vace_e." \
    --dataset_repeat 100 \
    --use_gradient_checkpointing_offload \
    --resume_path "/scratch/work/liz23/h2rego_video_generation/models/train/Wan2.1-VACE-E-1.3B_full/epoch-3.safetensors" \
    --resume_path_club "/scratch/work/liz23/h2rego_video_generation/models/train/Wan2.1-VACE-E-1.3B_full/epoch-3-club_estimator.safetensors"