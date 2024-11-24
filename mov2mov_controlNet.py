import cv2
import os

def extract_frames(video_path, frames_dir):
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
    
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    while success:
        frame_path = os.path.join(frames_dir, f"frame_{count:05d}.png")
        cv2.imwrite(frame_path, image)  # フレームを保存
        success, image = vidcap.read()
        count += 1
    vidcap.release()
    print(f"Extracted {count} frames.")

def create_video(frames_dir, output_video_path, fps=30):
    import moviepy.editor as mpy
    frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".png")])
    clip = mpy.ImageSequenceClip(frame_files, fps=fps)
    clip.write_videofile(output_video_path)
    print(f"Created video {output_video_path}.")


import torch
from diffusers import StableDiffusionImg2ImgPipeline, ControlNetModel, StableDiffusionControlNetImg2ImgPipeline
from PIL import Image
import numpy as np
import random

def load_controlnet_model(controlnet_model_name, device):
    controlnet = ControlNetModel.from_pretrained(
        controlnet_model_name,
        torch_dtype=torch.float16
    )
    return controlnet.to(device)

def initialize_pipeline(model_path, controlnet, device, scheduler_type="DPMSolver"):
    if controlnet:
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            model_path,
            controlnet=controlnet,
            torch_dtype=torch.float16
        ).to(device)
    else:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16
        ).to(device)
    
    # スケジューラーの設定
    if scheduler_type == "EulerAncestral":
        from diffusers import EulerAncestralDiscreteScheduler
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler_type == "DPMSolver":
        from diffusers import DPMSolverMultistepScheduler
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, final_sigmas_type="sigma_min")
    
    return pipe


from torchvision import transforms

def imgToImg_with_reference(
    pipe,
    img_path,
    prompt,
    output_path,
    device='cuda',
    guidance_scale=7,
    strength=0.8,
    num_inference_steps=40,
    seed=136,
    negative_prompt=None,
    reference_image=None
):
    # 画像の読み込み
    img0 = Image.open(img_path).convert("RGB")
    
    # 参照画像が指定されている場合、ControlNetを使用
    if reference_image:
        # 参照画像の前処理（例: エッジ検出）
        import cv2
        img_np = np.array(reference_image)
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(img_gray, 80, 150)
        edges = np.stack([edges]*3, axis=2)
        control_image = Image.fromarray(edges)
    else:
        control_image = None
    
    # ジェネレーターのシード設定
    generator = torch.Generator(device).manual_seed(seed)
    
    # 画像生成
    if reference_image and control_image:
        img = pipe(
            prompt=prompt,
            image=img0,
            control_image=control_image,
            guidance_scale=guidance_scale,
            strength=strength,
            num_inference_steps=num_inference_steps,
            generator=generator,
            negative_prompt=negative_prompt
        ).images[0]
    else:
        img = pipe(
            prompt=prompt,
            image=img0,
            guidance_scale=guidance_scale,
            strength=strength,
            num_inference_steps=num_inference_steps,
            generator=generator,
            negative_prompt=negative_prompt
        ).images[0]
    
    # 画像の保存
    img.save(output_path)
    
    denoise_image(img_path,output_path)
    np_img = np.array(img)
    
    return np_img

def denoise_image(image_path, output_path):
    import cv2
    img = cv2.imread(image_path)
    denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    cv2.imwrite(output_path, denoised)

def mov2mov(
    input_video_path,
    output_video_path,
    model_path,
    controlnet_model_name,
    prompt,
    output_frames_dir="output_frames",
    temp_frames_dir="temp_frames",
    device='cuda',
    scheduler_type="DPMSolver",
    guidance_scale=7,
    strength=0.6,
    num_inference_steps=40,
    seed=134,
    negative_prompt=None
):
    # フレームの抽出
    extract_frames(input_video_path, temp_frames_dir)
    
    # ControlNetのロード
    controlnet = load_controlnet_model(controlnet_model_name, device)
    
    # パイプラインの初期化
    pipe = initialize_pipeline(model_path, controlnet, device, scheduler_type)
    
    # 最初のフレームを参照画像として使用
    reference_frame_path = os.path.join(temp_frames_dir, "frame_00000.png")
    reference_image = Image.open(reference_frame_path).convert("RGB")
    
    # 変換されたフレームを保存するディレクトリを作成
    if not os.path.exists(output_frames_dir):
        os.makedirs(output_frames_dir)
    
    # フレームのリストを取得
    frame_files = sorted([f for f in os.listdir(temp_frames_dir) if f.endswith(".png")])
    
    for frame_file in frame_files:
        frame_path = os.path.join(temp_frames_dir, frame_file)
        output_frame_path = os.path.join(output_frames_dir, frame_file)
        
        # 各フレームを変換
        imgToImg_with_reference(
            pipe=pipe,
            img_path=frame_path,
            prompt=prompt,
            output_path=output_frame_path,
            device=device,
            guidance_scale=guidance_scale,
            strength=strength,
            num_inference_steps=num_inference_steps,
            seed=seed,
            negative_prompt=negative_prompt,
            reference_image=reference_image  # 参照画像を使用
        )
        
        print(f"Processed {frame_file}")
    
    # 動画の再構築
    create_video(output_frames_dir, output_video_path)
    
    # 一時フレームの削除（オプション）
    import shutil
    shutil.rmtree(temp_frames_dir)
    print("mov2mov processing completed.")



if __name__ == "__main__":
    input_video = "ドット.mp4"  # 入力動画のパス
    output_video = "output.mp4"  # 出力動画のパス
    model_path = "runwayml/stable-diffusion-v1-5" # Stable Diffusionモデルのパス
    controlnet_model = "lllyasviel/sd-controlnet-canny"  # ControlNetのモデル名（Cannyエッジ検出）
    prompt = "(Cinematic Aesthetic:1.4) Realistic photo, a cute girl, dance , moving, dynamic, girl wearing a blue hat, Long Sleeve Clothes,4k, white color heir"  # 生成に使用するプロンプト
    
    mov2mov(
        input_video_path=input_video,
        output_video_path=output_video,
        model_path=model_path,
        controlnet_model_name=controlnet_model,
        prompt=prompt,
        device='cuda',
        scheduler_type="DPMSolver",
        guidance_scale=7,
        strength=0.6,
        num_inference_steps=40,
        seed=134,
        negative_prompt="worst quality,lowers,pixel art,illustration,painting,cartoons,sketch, change pose"
    )
