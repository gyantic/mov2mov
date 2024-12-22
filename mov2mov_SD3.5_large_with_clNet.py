#動作確認ができていないファイルです。

from moviepy.editor import VideoFileClip, ImageSequenceClip
from diffusers import StableDiffusion3Img2ImgPipeline
from diffusers import StableDiffusion3ControlNetPipeline
from diffusers import SD3ControlNetModel, StableDiffusion3ControlNetPipeline
import torch
from transformers import BitsAndBytesConfig
import cv2
import os
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as F
import numpy as np
from diffusers.image_processor import VaeImageProcessor

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class SD3CannyImageProcessor(VaeImageProcessor):
    def __init__(self):
        super().__init__(do_normalize=False)
    def preprocess(self, image, **kwargs):
        image = super().preprocess(image, **kwargs)
        image = image * 255 * 0.5 + 0.5
        return image
    def postprocess(self, image, do_denormalize=True, **kwargs):
        do_denormalize = [True] * image.shape[0]
        image = super().postprocess(image, **kwargs, do_denormalize=do_denormalize)
        return image

controlnet = SD3ControlNetModel.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large-controlnet-canny",
    torch_dtype=torch.float16,
    joint_attention_dim=4096,
    low_cpu_mem_usage=False,
    device_map=None
)

controlnet_pipeline = StableDiffusion3ControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large",
    controlnet=controlnet,
    torch_dtype=torch.float16
)

controlnet_pipeline.image_processor = SD3CannyImageProcessor()

    # メモリ効率化の設定
controlnet_pipeline.enable_attention_slicing()

img2img_pipeline = StableDiffusion3Img2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large",
    torch_dtype=torch.float16
).to("cuda")

    # メモリ効率化の設定
img2img_pipeline.enable_attention_slicing()



# 動画の分割
def split_video_to_frames(video_path, output_dir):
    clip = VideoFileClip(video_path)
    os.makedirs(output_dir, exist_ok=True)
    for i, frame in enumerate(clip.iter_frames()):
        cv2.imwrite(f"{output_dir}/frame_{i:04d}.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

# Edge Detection (cannyで)
def edge_detection(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(image, 100, 200)
    return edges

# img2imgで各フレームに対してスタイル適用
def apply_img2img_with_reference(frame_path, edges_image, controlnet_pipeline, img2img_pipeline):
    # オリジナルの解像度を取得
    original_image = Image.open(frame_path).convert("RGB")
    width, height = original_image.size

    # 幅と高さを64の倍数に調整
    original_image = original_image.resize((512, 512))
    edges_image = edges_image.resize((512, 512))

    edges_image = edges_image.convert("RGB")
    # 画像生成
    controlnet_image = controlnet_pipeline(
        prompt="(Cinematic Aesthetic:1.4) Realistic photo, a cowboy, dance , moving, dynamic, man wearing a brown hat, Long Sleeve Clothes,4k",
        negative_prompt="cartoon, lowres, blurry, pixelated, sketch, drawing, NSFW, nude, naked, porn, ugly",
        #image=edges_image,
        controlnet_conditioning_scale=0.6,
        num_inference_steps=20, # 推論ステップ数を調整
    ).images[0]

    # 3. img2img で ControlNet の出力を利用して生成
    original_image = Image.open(frame_path).convert("RGB").resize((512, 512)) # 元画像

    generated_image = img2img_pipeline(
        prompt="(Cinematic Aesthetic:1.4) Realistic photo, a cowboy, dance , moving, dynamic, man wearing a brown hat, Long Sleeve Clothes,4k",
        negative_prompt="cartoon, lowres, blurry, pixelated, sketch, drawing, NSFW, nude, naked, porn, ugly",
        image=original_image, # 元画像を入力
        strength=0.6,  # img2img の強度を調整
        guidance_scale=10,
        num_inference_steps=20, # 推論ステップ数を調整
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


def mov2mov(video_path,output_dir, output_video_path):
    split_video_to_frames(video_path, output_dir)

    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    # メインモデルのパス
    model_path = "stabilityai/stable-diffusion-3.5-large"


    img2img_pipeline.image_processor = SD3CannyImageProcessor()

    # メモリ効率化の設定
    img2img_pipeline.enable_attention_slicing()
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

        # エッジ検出
        edges = edge_detection(frame_path)
        edges_image = Image.fromarray(edges).convert("RGB")

        # 画像生成（入力サイズを維持）
        styled_frame = apply_img2img_with_reference(frame_path, edges_image, controlnet_pipeline, img2img_pipeline)
        styled_frame.save(os.path.join(styled_frames_dir, frame_name))

    # 動画の再構築
    combine_frames_to_video(styled_frames_dir, output_video_path)