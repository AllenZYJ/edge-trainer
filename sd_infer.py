from transformers import CLIPTokenizer, CLIPTextModel
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from models.SD_simple import SimpleStableDiffusionModel
from trainers.sd_trainer import sd_trainer
import torch.nn.functional as F
if __name__ == "__main__":
    # 加载模型inference
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SimpleStableDiffusionModel()
    model.load_state_dict(torch.load("stable_diffusion_model.pth"))
    model.to(device)
    model.eval()  # 切换到评估模式
    model = SimpleStableDiffusionModel()
    text = "A box."
    save_path = "generated_image.png"
    model.inference_and_save(model, text, save_path)