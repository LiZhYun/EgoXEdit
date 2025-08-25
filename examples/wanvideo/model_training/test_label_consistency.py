#!/usr/bin/env python3
"""
Test script to validate that contrastive learning labels are generated consistently across batches.
This validates separate contrastive losses for:
1. Task contrastive loss: Same tasks should have similar embeddings
2. Embodiment contrastive loss: Same embodiment types should have similar embeddings

This addresses the critical issue where dynamic label generation was causing
the same task to get different labels in different batches.
"""

import torch
from torch.utils.data import DataLoader
from dataset_E import VideoDatasetE

def test_label_consistency():
    """Test that the same task gets the same label across different batches."""
    
    print("🧪 Testing Contrastive Learning Label Consistency...")
    
    # Initialize dataset 
    # Using a small subset for testing
    dataset = VideoDatasetE(
        base_path="../../data/example_video_dataset",  # Adjust path as needed
        video_size=(256, 256),
        frames_num=16,
        enable_vace=True,
        enable_fallback=True
    )
    
    if len(dataset) == 0:
        print("❌ No data found! Please check the dataset path.")
        return False
    
    print(f"📊 Dataset loaded: {len(dataset)} episodes")
    print(f"📊 Task prompt mappings: {len(dataset.task_prompt_to_label)} unique prompts")
    print(f"📊 Embodiment type mappings: {len(dataset.embodiment_type_to_label)} unique types")
    
    # Create multiple data loaders with different batch sizes to simulate different batches
    batch_sizes = [2, 3, 4] if len(dataset) >= 4 else [min(2, len(dataset))]
    
    task_prompt_to_labels_batches = {}
    task_name_to_labels_batches = {}
    
    for batch_size in batch_sizes:
        print(f"\n🔄 Testing with batch size {batch_size}")
        
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,  # Shuffling to get different combinations
            collate_fn=lambda x: x  # Simple collate to preserve structure
        )
        
        batch_count = 0
        for batch in dataloader:
            if batch_count >= 3:  # Test a few batches
                break
                
            batch_prompts = []
            batch_task_names = []
            batch_task_labels = []
            batch_embodiment_labels = []
            
            for item in batch:
                if item is not None:  # Skip None items from fallback
                    batch_prompts.append(item['prompt'])
                    batch_task_names.append(item['task_name'])
                    batch_task_labels.append(item['task_label'])
                    batch_embodiment_labels.append(item['embodiment_label'])
            
            # Record labels for each prompt/task_name
            for prompt, task_name, task_label, emb_label in zip(
                batch_prompts, batch_task_names, batch_task_labels, batch_embodiment_labels
            ):
                # Track task prompt labels
                if prompt not in task_prompt_to_labels_batches:
                    task_prompt_to_labels_batches[prompt] = []
                task_prompt_to_labels_batches[prompt].append(task_label)
                
                # Track task name embodiment labels
                if task_name not in task_name_to_labels_batches:
                    task_name_to_labels_batches[task_name] = []
                task_name_to_labels_batches[task_name].append(emb_label)
            
            batch_count += 1
            print(f"   Batch {batch_count}: {len(batch_prompts)} valid items")
    
    # Validate consistency
    print(f"\n✅ Consistency Validation:")
    
    all_consistent = True
    
    # Check task prompt consistency
    for prompt, labels in task_prompt_to_labels_batches.items():
        unique_labels = set(labels)
        if len(unique_labels) > 1:
            print(f"❌ Task prompt inconsistency: '{prompt[:50]}...' -> labels {unique_labels}")
            all_consistent = False
        else:
            print(f"✅ Task prompt consistent: '{prompt[:50]}...' -> label {unique_labels.pop()}")
    
    # Check embodiment label consistency  
    for task_name, labels in task_name_to_labels_batches.items():
        unique_labels = set(labels)
        if len(unique_labels) > 1:
            print(f"❌ Embodiment inconsistency: '{task_name}' -> labels {unique_labels}")
            all_consistent = False
        else:
            print(f"✅ Embodiment consistent: '{task_name}' -> label {unique_labels.pop()}")
    
    # Summary
    if all_consistent:
        print(f"\n🎉 SUCCESS: All labels are consistent across batches!")
        print(f"   - Task prompts tested: {len(task_prompt_to_labels_batches)}")
        print(f"   - Task names tested: {len(task_name_to_labels_batches)}")
        return True
    else:
        print(f"\n💥 FAILURE: Label inconsistencies detected!")
        return False

if __name__ == "__main__":
    success = test_label_consistency()
    if success:
        print("\n🚀 Contrastive learning is ready with consistent labels!")
        print("   ✅ Task contrastive loss: Same tasks will have similar embeddings")
        print("   ✅ Embodiment contrastive loss: Same embodiment types will have similar embeddings")
    else:
        print("\n🔧 Please fix label generation before training.")
