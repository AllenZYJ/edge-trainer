import torch
from torch import nn
from diffusers import UNet2DConditionModel, DDPMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
import numpy as np
from PIL import Image
class SimpleStableDiffusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义UNet模型
        self.unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
        # 定义VAE(编码器-解码器)
        self.vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
        # 定义文本编码模型
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        # 定义调度器(用于噪声处理)
        self.scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
    
    def forward(self, input_ids, pixel_values):
        # 对文本进行编码
        text_embeddings = self.text_encoder(input_ids).last_hidden_state
        # 将图像压缩到潜空间
        latents = self.vae.encode(pixel_values).latent_dist.sample()
        latents = 0.18215 * latents
        # 在潜空间中加入噪声
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        # 使用UNet进行噪声预测
        noise_pred = self.unet(noisy_latents, timesteps, text_embeddings).sample
        return noise_pred, noise,timesteps, latents
    def inference_and_save(self,model, text, save_path):
        # Tokenize the input text
        input_ids = model.tokenizer(text, return_tensors="pt").input_ids
        pixel_values = torch.randn(1, 3, 512, 512)  # 用随机噪声初始化潜在变量
        
        # Run the model
        noise_pred, noise, timesteps, latents = model(input_ids, pixel_values)
        
        # 反向调度以生成图像
        scheduler_output = model.scheduler.step(noise_pred, timesteps, latents)
        latents = scheduler_output['prev_sample']
        latents = latents / 0.18215  # 反向缩放
        
        # 使用VAE解码生成图像
        image = model.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)  # 将像素值归一化到[0,1]
        image = image.cpu().detach().numpy()
        image = np.transpose(image, (0, 2, 3, 1))  # 调整维度顺序为 (H, W, C)
        
        # 保存图像
        image_pil = Image.fromarray((image[0] * 255).astype(np.uint8))
        image_pil.save(save_path)