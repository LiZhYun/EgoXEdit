#!/usr/bin/env python3
"""
Test script for VideoDatasetE to validate the redesigned dataset.

Usage:
    python test_dataset_E.py
"""

import sys
import os
sys.path.append('/home/zhiyuan/Codes/DiffSynth-Studio')

from dataset_E import VideoDatasetE
import argparse


def test_dataset_discovery():
    """Test basic dataset discovery and validation."""
    print("=" * 60)
    print("🧪 Testing VideoDatasetE Dataset Discovery")
    print("=" * 60)
    
    try:
        # Create dataset with default paths
        dataset = VideoDatasetE(
            base_path="/home/zhiyuan/Codes/DataSets/small_test",
            task_metadata_path="/home/zhiyuan/Codes/human-policy/data/ph2d_metadata.json",
            num_frames=81,
            enable_fallback=True
        )
        
        print(f"✓ Dataset initialized successfully")
        print(f"✓ Found {len(dataset.episodes)} episodes")
        print(f"✓ Task metadata loaded: {len(dataset.task_metadata)} tasks")
        
        # Test sample loading
        if len(dataset) > 0:
            print(f"\n📋 Testing sample loading...")
            sample = dataset[0]
            
            if sample is not None:
                print(f"✓ Sample loaded successfully")
                print(f"  - Task: {sample.get('task_name')}")
                print(f"  - Episode: {sample.get('episode_name')}")
                print(f"  - Prompt: {sample.get('prompt')}")
                print(f"  - Target video: {len(sample.get('video', [])) if sample.get('video') else 'None'} frames")
                print(f"  - VACE control: {len(sample.get('vace_video', [])) if sample.get('vace_video') else 'None'} frames")
                print(f"  - VACE mask: {len(sample.get('vace_video_mask', [])) if sample.get('vace_video_mask') else 'None'} frames")
                print(f"  - Hand motion: {sample.get('hand_motion_sequence').shape if sample.get('hand_motion_sequence') is not None else 'None'}")
                print(f"  - Object trajectory: {sample.get('object_trajectory_sequence').shape if sample.get('object_trajectory_sequence') is not None else 'None'}")
                print(f"  - Object IDs: {sample.get('object_ids').shape if sample.get('object_ids') is not None else 'None'}")
                print(f"  - Embodiment image: {'Available' if sample.get('embodiment_image') else 'None'}")
                print(f"  - VACE reference: {'Available' if sample.get('vace_reference_image') else 'None'}")
                
                return True
            else:
                print("❌ Failed to load sample")
                return False
        else:
            print("⚠️ No episodes found - cannot test sample loading")
            return False
            
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_prompt_generation():
    """Test prompt generation from metadata."""
    print("\n" + "=" * 60)
    print("🧪 Testing Prompt Generation")
    print("=" * 60)
    
    try:
        dataset = VideoDatasetE(
            base_path="/home/zhiyuan/Codes/DataSets/small_test",
            task_metadata_path="/home/zhiyuan/Codes/human-policy/data/ph2d_metadata.json",
        )
        
        # Test known task names from metadata
        test_cases = [
            ("104-lars-grasping_2024-11-08_15-23-40", None),
            ("902-pouring-val-2024_11_18-18_49_25", None),
            ("unknown-task", None),
        ]
        
        for task_name, _ in test_cases:
            prompt = dataset.generate_task_prompt(task_name)
            print(f"  {task_name:<50} → {prompt}")
        
        print("✓ Prompt generation test completed")
        return True
        
    except Exception as e:
        print(f"❌ Prompt generation test failed: {e}")
        return False


def test_hdf5_structure():
    """Test HDF5 file structure understanding."""
    print("\n" + "=" * 60)
    print("🧪 Testing HDF5 Structure Understanding")
    print("=" * 60)
    
    try:
        dataset = VideoDatasetE(
            base_path="/home/zhiyuan/Codes/DataSets/small_test",
            task_metadata_path="/home/zhiyuan/Codes/human-policy/data/ph2d_metadata.json",
        )
        
        if len(dataset.episodes) > 0:
            episode = dataset.episodes[0]
            episode_path = episode['episode_path']
            episode_name = episode['episode_name']
            
            print(f"Testing episode: {episode_name}")
            
            # Test hand motion HDF5
            hand_hdf5_path = os.path.join(episode_path, f"{episode_name}_hand_trajectories.hdf5")
            if os.path.exists(hand_hdf5_path):
                try:
                    import h5py
                    with h5py.File(hand_hdf5_path, 'r') as f:
                        print(f"  Hand HDF5 keys: {list(f.keys())}")
                        if 'left_wrist' in f:
                            print(f"    left_wrist subkeys: {list(f['left_wrist'].keys())}")
                        if 'right_wrist' in f:
                            print(f"    right_wrist subkeys: {list(f['right_wrist'].keys())}")
                        
                    # Test loading
                    hand_motion = dataset.load_hand_motion_from_hdf5(hand_hdf5_path)
                    if hand_motion is not None:
                        print(f"  ✓ Hand motion loaded: {hand_motion.shape} (natural length, 20D format)")
                    else:
                        print(f"  ❌ Failed to load hand motion")
                except Exception as e:
                    print(f"  ❌ Hand HDF5 error: {e}")
            else:
                print(f"  ⚠️ Hand HDF5 not found: {hand_hdf5_path}")
            
            # Test object trajectory HDF5
            obj_hdf5_path = os.path.join(episode_path, f"{episode_name}_object_trajectories.hdf5")
            if os.path.exists(obj_hdf5_path):
                try:
                    import h5py
                    with h5py.File(obj_hdf5_path, 'r') as f:
                        print(f"  Object HDF5 keys: {list(f.keys())}")
                        object_groups = [key for key in f.keys() if key.startswith('object_')]
                        for obj_group in object_groups[:2]:  # Show first 2
                            print(f"    {obj_group} subkeys: {list(f[obj_group].keys())}")
                            if hasattr(f[obj_group], 'attrs'):
                                print(f"    {obj_group} attrs: {dict(f[obj_group].attrs)}")
                        
                    # Test loading
                    obj_traj, obj_ids = dataset.load_object_trajectory_from_hdf5(obj_hdf5_path)
                    if obj_traj is not None:
                        print(f"  ✓ Object trajectories loaded: {obj_traj.shape} (natural length, 9D per object)")
                        print(f"  ✓ Object IDs: {obj_ids}")
                    else:
                        print(f"  ❌ Failed to load object trajectories")
                except Exception as e:
                    print(f"  ❌ Object HDF5 error: {e}")
            else:
                print(f"  ⚠️ Object HDF5 not found: {obj_hdf5_path}")
            
            print("✓ HDF5 structure test completed")
            print("\n📝 Note: Dataset uses natural sequence lengths without padding.")
            print("   The VACE-E model handles variable lengths with attention masks,")
            print("   exactly like the inference pipeline (Wan2.1-VACE-1.3B_E.py).")
            return True
        else:
            print("⚠️ No episodes found for HDF5 testing")
            return False
            
    except Exception as e:
        print(f"❌ HDF5 structure test failed: {e}")
        return False


def test_with_args():
    """Test dataset creation with argparse args."""
    print("\n" + "=" * 60)
    print("🧪 Testing Args-based Initialization")
    print("=" * 60)
    
    try:
        # Create mock args
        args = argparse.Namespace()
        args.dataset_base_path = "/home/zhiyuan/Codes/DataSets/small_test"
        args.task_metadata_path = "/home/zhiyuan/Codes/human-policy/data/ph2d_metadata.json"
        args.height = 480
        args.width = 832
        args.max_pixels = 1920*1080
        args.num_frames = 81
        args.dataset_repeat = 1
        args.max_hand_motion_length = 512
        args.max_object_trajectory_length = 512
        args.max_objects = 10
        args.fallback_to_video_only = True
        
        dataset = VideoDatasetE(args=args)
        print(f"✓ Args-based initialization successful")
        print(f"  - Episodes: {len(dataset.episodes)}")
        print(f"  - Dynamic resolution: {dataset.dynamic_resolution}")
        print(f"  - Target size: {dataset.height}x{dataset.width}")
        
        return True
        
    except Exception as e:
        print(f"❌ Args-based test failed: {e}")
        return False


def main():
    """Run all dataset tests."""
    print("🚀 Starting VideoDatasetE Test Suite")
    
    tests = [
        test_dataset_discovery,
        test_prompt_generation, 
        test_hdf5_structure,
        test_with_args,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 All tests passed! Dataset is ready for training.")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the dataset setup.")
        return 1


if __name__ == "__main__":
    exit(main()) 