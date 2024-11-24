from moviepy.editor import VideoFileClip, ImageSequenceClip
from diffusers import StableDiffusion3Img2ImgPipeline, StableDiffusionImg2ImgPipeline
import torch
from transformers import T5EncoderModel, BitsAndBytesConfig
import cv2
import os
from PIL import Image
from tqdm import tqdm

# 動画の分割
def split_video_to_frames(video_path, output_dir):
    clip = VideoFileClip(video_path)
    os.makedirs(output_dir, exist_ok=True)
    for i, frame in enumerate(clip.iter_frames()):
        cv2.imwrite(f"{output_dir}/frame_{i:04d}.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

# img2imgで各フレームに対してスタイル適用
def apply_img2img_with_reference(frame_path, reference_image, pipeline):
    # オリジナルの解像度を取得
    original_image = Image.open(frame_path).convert("RGB")
    width, height = original_image.size

    # 幅と高さを64の倍数に調整
    w, h = map(lambda x: x - x % 64, (width, height))
    original_image = original_image.resize((w, h))
    #edges_image = edges_image.resize((w, h))
    reference_image_resized = reference_image.resize((w, h))

    # 画像生成
    with torch.no_grad():
        generated_image = pipeline(
            prompt="(Cinematic Aesthetic:1.4) Realistic photo, a cowboy, dance , moving, dynamic, man wearing a brown hat, Long Sleeve Clothes,4k",
            negative_prompt="cartoon, lowres, blurry, pixelated, sketch, drawing",
            image=original_image,
            guidance_scale=10,
            strength=0.6,
            num_inference_steps=40,
        ).images[0]

    return generated_image

# 動画の再構築
def combine_frames_to_video(frames_dir, output_video_path, fps=30):
    frame_files = sorted(
        [img for img in os.listdir(frames_dir) if os.path.isfile(os.path.join(frames_dir, img))],
        key=lambda x: int(os.path.splitext(x)[0].split('_')[-1])
    )
    frames = [cv2.imread(os.path.join(frames_dir, img)) for img in frame_files]
    clip = ImageSequenceClip([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames], fps=fps)
    clip.write_videofile(output_video_path, codec="libx264")


def mov2mov(video_path, reference_image_path, output_dir, output_video_path):
    split_video_to_frames(video_path, output_dir)
    reference_image = Image.open(reference_image_path).convert("RGB")

    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    # メインモデルのパス
    #model_path = "stabilityai/stable-diffusion-3-medium-diffusers"
    model_path = "stabilityai/stable-diffusion-3.5-medium"


    # パイプラインの初期化
    pipeline = StableDiffusion3Img2ImgPipeline.from_pretrained(
        model_path,
        text_encoder_3=None,
        tokenizer_3=None,
        torch_dtype=torch.float16,
    ).to("cuda")

    # メモリ効率化の設定
    pipeline.enable_attention_slicing()
    #pipeline.enable_model_cpu_offload()

    # スタイリング後のフレームを保存するディレクトリ
    styled_frames_dir = os.path.join(output_dir, "styled_frames")
    os.makedirs(styled_frames_dir, exist_ok=True)

    # フレームの処理
    frame_files = sorted(
        [img for img in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, img))],
        key=lambda x: int(os.path.splitext(x)[0].split('_')[-1])
    )

    for frame_name in tqdm(frame_files, desc="Processing frames"):
        frame_path = os.path.join(output_dir, frame_name)

        # 画像生成（入力サイズを維持）
        styled_frame = apply_img2img_with_reference(frame_path, reference_image, pipeline)
        styled_frame.save(os.path.join(styled_frames_dir, frame_name))

    # 動画の再構築
    combine_frames_to_video(styled_frames_dir, output_video_path)

mov2mov("SD記事用動画.mp4","reference_記事.jpg","frames","SD3.5__noNet.mp4")