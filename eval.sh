python examples/wanvideo/model_inference/Wan2.1-VACE-1.3B_E.py \
    --state_dict_path ./models/train/Wan2.1-VACE-E-1.3B_full/epoch-1.safetensors \
    --dataset_path ./data/PH2D_videos \
    --output_dir videos_club \
    --end_effector_image ./data/PH2D_videos/robot_hands.jpg \
    --vace_e_scale 1.0 \
