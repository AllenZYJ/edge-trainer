import torch
from torch import nn
from diffusers import UNet2DConditionModel, DDPMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer

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
        text_embeddings = self.text_encoder(input_ids)[0]
        # 将图像压缩到潜空间
        latents = self.vae.encode(pixel_values).latent_dist.sample()
        latents = 0.18215 * latents
        # 在潜空间中加入噪声
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        # 使用UNet进行噪声预测
        print(noisy_latents.shape)
        print(text_embeddings.shape)
        noise_pred = self.unet(noisy_latents, timesteps, text_embeddings).sample
        return noise_pred, noise