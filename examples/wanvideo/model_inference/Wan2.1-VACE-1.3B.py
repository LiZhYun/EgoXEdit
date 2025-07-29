import torch
from PIL import Image
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download
import os
import tempfile
import shutil


pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
    ],
)

pipe.enable_vram_management()

# dataset_snapshot_download(
#     dataset_id="DiffSynth-Studio/examples_in_diffsynth",
#     local_dir="./",
#     allow_file_pattern=["data/examples/wan/depth_video.mp4", "data/examples/wan/cat_fightning.jpg"]
# )

# # Depth video -> Video
control_video = VideoData("data/examples/wan/depth_video.mp4", height=480, width=832)
# video = pipe(
#     prompt="两只可爱的橘猫戴上拳击手套，站在一个拳击台上搏斗。",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
#     vace_video=control_video,
#     seed=1, tiled=True
# )
# save_video(video, "video1.mp4", fps=15, quality=5)

# # Reference image -> Video
# video = pipe(
#     prompt="两只可爱的橘猫戴上拳击手套，站在一个拳击台上搏斗。",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
#     vace_reference_image=Image.open("data/examples/wan/cat_fightning.jpg").resize((832, 480)),
#     seed=1, tiled=True
# )
# save_video(video, "video2.mp4", fps=15, quality=5)

# # Depth video + Reference image -> Video
# video = pipe(
#     prompt="两只可爱的橘猫戴上拳击手套，站在一个拳击台上搏斗。",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
#     vace_video=control_video,
#     vace_reference_image=Image.open("data/examples/wan/cat_fightning.jpg").resize((832, 480)),
#     seed=1, tiled=True
# )
# save_video(video, "video3.mp4", fps=15, quality=5)

# the videos are too long that the vram is not enough, so we need to reduce the frame count
# the frame count should be divisible by 4 + 1, like 17, 21, 81
# we can reduce the frame count to 81

vace_video = VideoData("data/examples/wan/masked_video_robot.mp4", height=480, width=832)
vace_video_mask = VideoData("data/examples/wan/mask_only_video_robot.mp4", height=480, width=832)

# Reduce frame count to 81 for VRAM efficiency
original_frames = len(vace_video)
target_frames = 81

if original_frames > target_frames:
    print(f"Reducing video from {original_frames} frames to {target_frames} frames")
    
    # Sample frames evenly from the video
    step = original_frames // target_frames
    sampled_indices = [i * step for i in range(target_frames)]
    
    # Extract sampled frames from both video and mask
    print("Extracting sampled frames...")
    vace_frames = [vace_video[i] for i in sampled_indices]
    mask_frames = [vace_video_mask[i] for i in sampled_indices]
    
    # Create temporary folders for sampled frames
    
    temp_dir = tempfile.mkdtemp()
    vace_temp_folder = os.path.join(temp_dir, "vace_frames")
    mask_temp_folder = os.path.join(temp_dir, "mask_frames")
    os.makedirs(vace_temp_folder, exist_ok=True)
    os.makedirs(mask_temp_folder, exist_ok=True)
    
    # Save sampled frames to temporary folders
    for i, (vace_frame, mask_frame) in enumerate(zip(vace_frames, mask_frames)):
        vace_frame.save(os.path.join(vace_temp_folder, f"{i:04d}.png"))
        mask_frame.save(os.path.join(mask_temp_folder, f"{i:04d}.png"))
    
    # Create new VideoData objects from sampled frames
    vace_video = VideoData(image_folder=vace_temp_folder, height=480, width=832)
    vace_video_mask = VideoData(image_folder=mask_temp_folder, height=480, width=832)
    
    print(f"✓ Reduced to {len(vace_video)} frames")
    
    # Clean up function (will be called after generation)
    def cleanup_temp_files():
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print("✓ Cleaned up temporary files")
else:
    print(f"Video already has {original_frames} frames (≤ {target_frames}), no reduction needed")
    def cleanup_temp_files():
        pass  # No cleanup needed

video = pipe(
    prompt="视频展示了两只手在桌面上操作。首先右手拿起瓶子，左手拿起水杯，然后右手左手靠近，将瓶子和水杯对准，右手倾斜瓶子，保持一段时间后，左右手各自放下手上的物体。整个视频的拍摄角度是固定的，重点展示了双机械臂的操作。",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    vace_video=vace_video,
    vace_video_mask=vace_video_mask,  # Start without mask for simpler debugging
    vace_reference_image=Image.open("data/examples/wan/masked_object.jpg").resize((832, 480)),
    seed=1, 
    tiled=True,
    height=480,
    width=832,
    num_frames=len(vace_video)  # Use the reduced frame count (81 or less)
)
save_video(video, "video2.mp4", fps=15, quality=5)

# Clean up temporary files
cleanup_temp_files()
print("✅ VACE video generation completed successfully!")
