# Multi-GPU CLUB Loss Implementation

## Problem
CLUB loss requires the full global batch for contrastive loss computation, but each GPU only has its local batch in multi-GPU training.

## Solution
Gather task and embodiment features from all GPUs before computing CLUB loss using `accelerator.gather()`.

## Key Changes

### 1. Modified `training_loss` function
- Added feature gathering using `accelerator.gather()`
- Fallback to `torch.distributed.all_gather()` if accelerator not available
- Fallback to local features if distributed training not available

### 2. Added `set_accelerator` method
- Allows pipeline to access accelerator instance
- Enables distributed feature gathering

### 3. Updated training module
- Added `set_accelerator` method to pass accelerator to pipeline
- Integrated with existing training loop

### 4. Updated training loop
- Set accelerator instance on model after preparation
- Enable distributed feature gathering

## Usage
```bash
python examples/wanvideo/model_training/train_E.py \
    --enable_club_loss \
    --batch_size 4 \
    --use_video_collate
```

## Testing
```bash
python test_multi_gpu_club.py --test_type gathering
``` 