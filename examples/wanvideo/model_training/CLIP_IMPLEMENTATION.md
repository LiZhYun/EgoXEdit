# Contrastive Learning - Consistent Label Implementation

## 🎯 Problem Solved

**Critical Issue**: The original implementation generated task and embodiment labels dynamically within each batch, causing the same task to receive different labels across different batches. This broke the fundamental assumption of contrastive learning where consistent semantics should have consistent labels.

**Root Cause**: Labels were generated in the loss function using dynamic per-batch label generation instead of consistent global mappings.

## ✅ Solution Implemented

**Separate Contrastive Losses**: Implemented two independent contrastive losses:
1. **Task Contrastive Loss**: Encourages same tasks to have similar embeddings, different tasks to have different embeddings
2. **Embodiment Contrastive Loss**: Encourages same embodiment types to have similar embeddings, different embodiment types to have different embeddings

**Global Label Generation**: Moved label creation to dataset initialization (`dataset_E.py`) to ensure consistency across all training batches.

### Key Changes

#### 1. Dataset Label Mapping Creation (`dataset_E.py`)
- Added `_create_label_mappings()` method that scans all episodes and creates consistent mappings
- `task_prompt_to_label`: Maps unique task prompts to consistent integer labels
- `embodiment_type_to_label`: Maps embodiment types ('human'/'robot') to consistent integer labels
- Added `get_task_label()` and `get_embodiment_label()` helper methods

#### 2. Data Item Updates (`dataset_E.py`)
Enhanced `__getitem__` to include pre-generated labels:
```python
data = {
    'task_name': task_name,
    'episode_name': episode_name,
    'prompt': task_prompt,
    # NEW: Pre-generated consistent labels
    'task_label': self.get_task_label(task_prompt),
    'embodiment_label': self.get_embodiment_label(task_name),
}
```

#### 3. Contrastive Loss Functions (`wan_video_new_E.py`)
- Replaced CLIP-style cross-modal contrastive loss with separate within-modality losses
- Added `compute_contrastive_loss()` for generic contrastive learning
- **Task Contrastive Loss**: Applied to task features with task labels
- **Embodiment Contrastive Loss**: Applied to embodiment features with embodiment labels
- Each loss encourages same-label features to be similar, different-label features to be different

#### 4. Training Integration (`wan_video_new_E.py`)
Updated training loop to compute both contrastive losses:
```python
# Task contrastive loss (same tasks should be similar)
task_contrastive_loss = self.compute_contrastive_loss(
    gathered_task_features, 
    gathered_task_labels,
    temperature=self.contrastive_temperature
)

# Embodiment contrastive loss (same embodiments should be similar)  
embodiment_contrastive_loss = self.compute_contrastive_loss(
    gathered_embodiment_features, 
    gathered_embodiment_labels,
    temperature=self.contrastive_temperature
)
```

## 🧪 Validation

### Test Script: `test_label_consistency.py`
Created comprehensive test to validate label consistency:
- Tests multiple batch sizes and shuffling
- Verifies same tasks get same labels across all batches
- Checks both task prompt labels and embodiment labels
- Provides detailed consistency reports

### Usage:
```bash
cd examples/wanvideo/model_training
python test_label_consistency.py
```

## 📊 Benefits

1. **Consistent Contrastive Learning**: Same tasks/embodiments always get same labels regardless of batch composition
2. **Better Task Representation**: Task features learn to cluster by semantic similarity
3. **Better Embodiment Representation**: Embodiment features learn to distinguish human vs robot
4. **Proper Independence**: Task and embodiment losses are computed separately without cross-modal confusion
5. **Multi-GPU Safe**: Labels are consistent across distributed training
6. **Debuggable**: Clear label mappings and validation tools

## 🚀 Training Integration

The contrastive losses now work seamlessly with existing CLUB loss:

```bash
python train_E.py \
    --base_path "../../data/example_video_dataset" \
    --enable_contrastive_loss \
    --contrastive_temperature 0.07 \
    --task_contrastive_lambda 1.0 \
    --embodiment_contrastive_lambda 1.0
```

## 🔍 Key Features

- **Separate Task/Embodiment Losses**: Independent contrastive learning for each modality
- **Global Consistency**: Labels generated once during dataset initialization  
- **Same-Class Attraction**: Features with same labels are encouraged to be similar
- **Different-Class Repulsion**: Features with different labels are pushed apart
- **Multi-GPU Support**: Proper feature gathering across devices
- **Validation Tools**: Built-in consistency checking and debugging

## 📋 Verification Checklist

- [x] Label mappings created during dataset initialization
- [x] Consistent labels added to data items
- [x] Separate contrastive loss functions implemented
- [x] Task contrastive loss for task feature clustering
- [x] Embodiment contrastive loss for embodiment feature clustering
- [x] Training integration updated
- [x] Training script arguments updated
- [x] Test script for validation updated
- [x] Documentation updated

The implementation now ensures that:
- **Same tasks learn similar representations** regardless of embodiment type
- **Same embodiment types learn similar representations** regardless of task
- **Labels remain consistent across all training batches** 🎉
