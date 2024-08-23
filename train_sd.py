import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import sys
sys.path.append("./")
from models.SD_simple import SimpleStableDiffusionModel
from trainers.sd_trainer import sd_trainer
import torch.nn.functional as F
# 图像转换
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # 调整图像大小
    transforms.ToTensor(),        # 转换为Tensor
])
train_dataset = ImageFolder(root='./datasets/simple_dataset/train', transform=transform)
val_dataset = ImageFolder(root='./datasets/simple_dataset/val', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
# 初始化模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SimpleStableDiffusionModel().to(device)
model.device = device
sd_trainer(model, train_loader, epochs=100, lr=1e-4)
# # 假设模型训练结束
torch.save(model.state_dict(), "stable_diffusion_model.pth")
