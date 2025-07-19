import os, json, requests, random, time, cv2, ffmpeg, runpod
from urllib.parse import urlsplit

import torch
from PIL import Image
import numpy as np

from nodes import NODE_CLASS_MAPPINGS
from comfy_extras import nodes_wan, nodes_model_advanced

UNETLoader = NODE_CLASS_MAPPINGS["UNETLoader"]()
CLIPLoader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
LoraLoaderModelOnly = NODE_CLASS_MAPPINGS["LoraLoaderModelOnly"]()
VAELoader = NODE_CLASS_MAPPINGS["VAELoader"]()

CLIPTextEncode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
LoadImage = NODE_CLASS_MAPPINGS["LoadImage"]()
ImageBatch = NODE_CLASS_MAPPINGS["ImageBatch"]()

WanPhantomSubjectToVideo = nodes_wan.NODE_CLASS_MAPPINGS["WanPhantomSubjectToVideo"]()
ConditioningCombine = NODE_CLASS_MAPPINGS["ConditioningCombine"]()

KSampler = NODE_CLASS_MAPPINGS["KSampler"]()
ModelSamplingSD3 = nodes_model_advanced.NODE_CLASS_MAPPINGS["ModelSamplingSD3"]()
VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()

with torch.inference_mode():
    unet = UNETLoader.load_unet("Phantom-Wan-14B_fp8_e4m3fn.safetensors", "default")[0]
    clip = CLIPLoader.load_clip("umt5_xxl_fp8_e4m3fn_scaled.safetensors", "wan")[0]
    lora = LoraLoaderModelOnly.load_lora_model_only(unet, "FusionX/Phantom_Wan_14B_FusionX_LoRA.safetensors", 1.0)[0]
    vae = VAELoader.load_vae("wan_2.1_vae.safetensors")[0]

def download_file(url, save_dir, file_name):
    os.makedirs(save_dir, exist_ok=True)
    file_suffix = os.path.splitext(urlsplit(url).path)[1]
    file_name_with_suffix = file_name + file_suffix
    file_path = os.path.join(save_dir, file_name_with_suffix)
    response = requests.get(url)
    response.raise_for_status()
    with open(file_path, 'wb') as file:
        file.write(response.content)
    return file_path

def images_to_mp4(images, output_path, fps=24):
    try:
        frames = []
        for image in images:
            i = 255. * image.cpu().numpy()
            img = np.clip(i, 0, 255).astype(np.uint8)
            if img.shape[0] in [1, 3, 4]:
                img = np.transpose(img, (1, 2, 0))
            if img.shape[-1] == 4:
                img = img[:, :, :3]
            frames.append(img)
        temp_files = [f"temp_{i:04d}.png" for i in range(len(frames))]
        for i, frame in enumerate(frames):
            success = cv2.imwrite(temp_files[i], frame[:, :, ::-1])
            if not success:
                raise ValueError(f"Failed to write {temp_files[i]}")
        if not os.path.exists(temp_files[0]):
            raise FileNotFoundError("Temporary PNG files were not created")
        stream = ffmpeg.input('temp_%04d.png', framerate=fps)
        stream = ffmpeg.output(stream, output_path, vcodec='libx264', pix_fmt='yuv420p')
        ffmpeg.run(stream, overwrite_output=True)
        for temp_file in temp_files:
            os.remove(temp_file)
    except Exception as e:
        print(f"Error: {e}")

@torch.inference_mode()
def generate(input):
    try:
        values = input["input"]

        input_image1 = values['input_image1']
        input_image1 = download_file(url=input_image1, save_dir='/content/dev/ComfyUI/input', file_name='input_image1')
        input_image2 = values['input_image2']
        input_image2 = download_file(url=input_image2, save_dir='/content/dev/ComfyUI/input', file_name='input_image2')
        input_image3 = values['input_image3']
        input_image3 = download_file(url=input_image3, save_dir='/content/dev/ComfyUI/input', file_name='input_image3')
        input_image4 = values['input_image4']
        input_image4 = download_file(url=input_image4, save_dir='/content/dev/ComfyUI/input', file_name='input_image4')
        positive_prompt = values['positive_prompt'] # A mid-shot of a asian woman in a sparkly pink crop top and low-rise cargo pants, dancing sharply in sync with the beat while singing straight into the camera. Her hair is styled in voluminous waves with front strands pulled into mini pigtails. Behind her, colored spotlights flash across a silver sequin curtain backdrop. Pure early-2000s pop performance.
        negative_prompt = values['negative_prompt'] # 色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿
        width = values['width'] # 1024
        height = values['height'] # 576
        length = values['length'] # 121
        batch_size = values['batch_size'] # 1
        shift = values['shift'] # 8.0
        cfg = values['cfg'] # 1.0
        sampler_name = values['sampler_name'] # uni_pc
        scheduler = values['scheduler'] # simple
        steps = values['steps'] # 8
        seed = values['seed'] # 1.0
        if seed == 0:
            random.seed(int(time.time()))
            seed = random.randint(0, 18446744073709551615)
        fps = values['fps'] # 24
        filename_prefix = values['filename_prefix'] # wan_i2v_phantom

        positive = CLIPTextEncode.encode(clip, positive_prompt)[0]
        negative = CLIPTextEncode.encode(clip, negative_prompt)[0]
        model = ModelSamplingSD3.patch(lora, shift)[0]
        input_image1 = LoadImage.load_image(input_image1)[0]
        input_image2 = LoadImage.load_image(input_image2)[0]
        input_image3 = LoadImage.load_image(input_image3)[0]
        input_image4 = LoadImage.load_image(input_image4)[0]
        input_image12 = ImageBatch.batch(input_image1, input_image2)[0]
        input_image123 = ImageBatch.batch(input_image12, input_image3)[0]
        input_images = ImageBatch.batch(input_image123, input_image4)[0]
        positive, negative_text, negative_img_text, latent_image = WanPhantomSubjectToVideo.encode(positive, negative, vae, width, height, length, batch_size, input_images)
        negative_combined = ConditioningCombine.combine(negative_text, negative_img_text)[0]
        samples = KSampler.sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative_combined, latent_image)[0]
        decoded_images = VAEDecode.decode(vae, samples)[0].detach()
        images_to_mp4(decoded_images, f"/content/wan2.1-i2v-phantom-{seed}-tost.mp4", fps)
        
        result = f"/content/wan2.1-i2v-phantom-{seed}-tost.mp4"

        notify_uri = values['notify_uri']
        del values['notify_uri']
        notify_token = values['notify_token']
        del values['notify_token']
        discord_id = values['discord_id']
        del values['discord_id']
        if(discord_id == "discord_id"):
            discord_id = os.getenv('com_camenduru_discord_id')
        discord_channel = values['discord_channel']
        del values['discord_channel']
        if(discord_channel == "discord_channel"):
            discord_channel = os.getenv('com_camenduru_discord_channel')
        discord_token = values['discord_token']
        del values['discord_token']
        if(discord_token == "discord_token"):
            discord_token = os.getenv('com_camenduru_discord_token')
        job_id = values['job_id']
        del values['job_id']
        with open(result, 'rb') as file:
            response = requests.post("https://upload.tost.ai/api/v1", files={'file': file})
        response.raise_for_status()
        result_url = response.text
        notify_payload = {"jobId": job_id, "result": result_url, "status": "DONE"}
        web_notify_uri = os.getenv('com_camenduru_web_notify_uri')
        web_notify_token = os.getenv('com_camenduru_web_notify_token')
        if(notify_uri == "notify_uri"):
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
        else:
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            requests.post(notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        return {"jobId": job_id, "result": result_url, "status": "DONE"}
    except Exception as e:
        error_payload = {"jobId": job_id, "status": "FAILED"}
        try:
            if(notify_uri == "notify_uri"):
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            else:
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
                requests.post(notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        except:
            pass
        return {"jobId": job_id, "result": f"FAILED: {str(e)}", "status": "FAILED"}
    finally:
        if os.path.exists(result):
            os.remove(result)

runpod.serverless.start({"handler": generate})