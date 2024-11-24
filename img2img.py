import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline, ControlNetModel, StableDiffusionControlNetImg2ImgPipeline
from translate import Translator
from diffusers import EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
import numpy as np
from transformers import CLIPTokenizer


modelFile1 = "E:/stadifmodels/beautifulRealistic_v7.safetensors"
modelFile2 = "E:/stadifmodels/rmadaMergeSD21768_v70.safetensors"
modelFile3 = "E:/stadifmodels/realmixpony_rev04fix.safetensors"

def load_controlnet_model(controlnet_model_name, device):
    controlnet = ControlNetModel.from_pretrained(
        controlnet_model_name,
        torch_dtype=torch.float16
    )
    return controlnet.to(device)




def imgToImg(
    model_path,
    img_path,
    prompt,
    output_path,
    device='cuda',
    scheduler_type="DPMSolver",
    guidance_scale=7,
    strength=0.6,
    num_inference_steps=40,
    seed=134,
    negative_prompt=None,
):
    # モデルのロード
    pipe = StableDiffusionImg2ImgPipeline.from_single_file(
        model_path,
        torch_dtype=torch.float16
    ).to(device)

    #pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16).to("cuda")


    # スケジューラーの設定
    if scheduler_type == "EulerAncestral":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler_type == "DPMSolver":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config,final_sigmas_type="sigma_min")
    
    # ジェネレーターのシード設定
    generator = torch.Generator(device).manual_seed(seed)

    # 画像の読み込み
    img0 = Image.open(img_path).convert("RGB")


    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")

    # プロンプトのトークン化
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # 翻訳機能（必要ならば使用）
    honyaku = Translator('en', 'ja').translate

    # 画像生成
    img = pipe(
        prompt,
        image=img0,
        guidance_scale=guidance_scale,
        strength=strength,
        num_inference_steps=num_inference_steps,
        generator=generator,
        negative_prompt=negative_prompt
    ).images[0]

    # 画像の保存
    img.save(output_path)
    np_img = np.array(img)

    return np_img

imgToImg(modelFile1,"article_sample.png",'(Cinematic Aesthetic:1.4) Realistic photo, a cauboy, dance , wearing a  hat, Long Sleeve Clothes,4k',"reference_記事.jpg",negative_prompt="worst quality,lowers,illustration,painting,cartoons,sketch, change pose")
 