# Multi-GPU CLUB Loss Implementation

This document describes the implementation of multi-GPU support for CLUB (Contrastive Log-ratio Upper Bound) loss in the VACE-E training pipeline.

## Problem Statement

The CLUB loss is a contrastive loss that requires the entire global batch to be present for proper mutual information estimation. In multi-GPU training scenarios, each GPU only has access to its local batch, which is insufficient for contrastive loss computation.

## Solution

The solution implements feature gathering across all GPUs before computing the CLUB loss, ensuring that each GPU has access to the full global batch for contrastive loss computation.

## Implementation Details

### 1. Feature Gathering in `training_loss`

The `training_loss` function in `diffsynth/pipelines/wan_video_new_E.py` has been modified to:

- Gather task and embodiment features from all GPUs using `accelerator.gather()`
- Fall back to distributed gather (`torch.distributed.all_gather`) if accelerator is not available
- Fall back to local features if neither is available

```python
# Gather features from all GPUs for contrastive loss computation
if hasattr(self, 'accelerator') and self.accelerator is not None:
    gathered_task_features = self.accelerator.gather(task_reduced)
    gathered_embodiment_features = self.accelerator.gather(embodiment_reduced)
```

### 2. Accelerator Integration

The pipeline now supports setting an accelerator instance:

```python
def set_accelerator(self, accelerator):
    """Set the accelerator instance for distributed training."""
    self.accelerator = accelerator
```

### 3. Training Module Updates

The `WanTrainingModuleE` class has been updated to:

- Pass the accelerator instance to the pipeline
- Support distributed feature gathering

```python
def set_accelerator(self, accelerator):
    """Set the accelerator instance for the pipeline."""
    self.pipe.set_accelerator(accelerator)
```

### 4. Training Loop Integration

The `launch_training_task` function in `diffsynth/trainers/utils.py` has been updated to:

- Set the accelerator instance on the model after preparation
- Enable distributed feature gathering

```python
# Set accelerator instance on model for distributed feature gathering
if hasattr(model, 'set_accelerator'):
    model.set_accelerator(accelerator)
```

## Usage

### Training with Multi-GPU CLUB Loss

1. **Enable CLUB Loss**: Set `--enable_club_loss` in your training script
2. **Configure Batch Size**: Ensure `batch_size > 1` for proper mutual information estimation
3. **Use Video Collate**: Set `--use_video_collate` for proper video batching

```bash
python examples/wanvideo/model_training/train_E.py \
    --enable_club_loss \
    --batch_size 4 \
    --use_video_collate \
    --club_lambda 1.0 \
    --club_update_freq 1 \
    --club_training_steps 5 \
    --club_lr 1e-3
```

### Testing Multi-GPU Feature Gathering

Use the provided test script to verify feature gathering functionality:

```bash
# Test feature gathering
python test_multi_gpu_club.py --test_type gathering

# Test CLUB loss simulation
python test_multi_gpu_club.py --test_type club
```

## Key Features

### 1. Automatic Fallback Strategy

The implementation includes a robust fallback strategy:

1. **Primary**: Use `accelerator.gather()` if accelerator is available
2. **Secondary**: Use `torch.distributed.all_gather()` if distributed training is available
3. **Fallback**: Use local features if neither is available

### 2. Error Handling

Comprehensive error handling ensures training continues even if feature gathering fails:

```python
except Exception as e:
    print(f"Warning: Error during feature gathering ({e}), using local features for CLUB loss")
    task_flat = task_reduced.to(dtype=self.torch_dtype)
    embodiment_flat = embodiment_reduced.to(dtype=self.torch_dtype)
```

### 3. Memory Efficiency

- Features are reduced using global average pooling before gathering
- Only necessary features are gathered across GPUs
- Proper dtype conversion for memory efficiency

## Configuration

### CLUB Loss Parameters

- `club_lambda`: Weight for CLUB loss in total loss (default: 1.0)
- `club_update_freq`: Update CLUB estimator every N training steps (default: 1)
- `club_training_steps`: Number of CLUB training steps per update (default: 5)
- `club_lr`: Learning rate for CLUB optimizer (default: 1e-3)

### Multi-GPU Training Parameters

- `batch_size`: Must be > 1 for proper CLUB loss computation
- `use_video_collate`: Required for proper video batching
- `gradient_accumulation_steps`: Configure for effective batch size

## Performance Considerations

### Memory Usage

- Feature gathering increases memory usage on each GPU
- Consider reducing feature dimensions if memory is limited
- Use gradient checkpointing for large models

### Communication Overhead

- Feature gathering adds communication overhead
- Overhead is minimal compared to model parameter synchronization
- Only occurs when CLUB loss is enabled

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce batch size or feature dimensions
2. **Communication Errors**: Check distributed training setup
3. **CLUB Loss Not Computing**: Ensure batch_size > 1 and enable_club_loss=True

### Debug Information

The implementation includes detailed logging:

```
Gathered feature shapes (accelerator): task=torch.Size([8, 2048]), embodiment=torch.Size([8, 2048])
🎯 CLUB loss configured: lambda=1.0, update_freq=1, training_steps=5, lr=0.001, enabled=True
```

## Future Improvements

1. **Selective Gathering**: Only gather features when CLUB loss is enabled
2. **Compression**: Implement feature compression for reduced communication
3. **Asynchronous Gathering**: Implement asynchronous feature gathering for better performance
4. **Dynamic Batch Size**: Automatically adjust batch size based on available memory

## References

- CLUB Loss: [Contrastive Log-ratio Upper Bound](https://arxiv.org/abs/2006.12013)
- Accelerate: [Hugging Face Accelerate](https://huggingface.co/docs/accelerate/)
- Distributed Training: [PyTorch Distributed](https://pytorch.org/docs/stable/distributed.html) 