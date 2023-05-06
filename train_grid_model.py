import torch
from torch import nn
from torch.utils.data import DataLoader
from models.grid_net import gridnet,ResNet,BasicBlock
import random
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Set seed.
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(seed)
random.seed(seed)
# 生成模拟数据
input_size = 640
output_size = 1080

x = torch.rand(100, 3, input_size, input_size)  
y = torch.randint(0,2,(100,1))
# 定义数据集
dataset = torch.utils.data.TensorDataset(x, y)
# 定义数据加载器
train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(dataset, batch_size=1)
# 模型,优化器和损失函数
# model = gridnet(batchsize=1)
model = ResNet(img_channels=3, num_layers=18, block=BasicBlock, num_classes=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()
# 训练模型 
from trainers.grid_trainer import grid_trainer
trainer = grid_trainer(model, optimizer, loss_fn, train_loader, val_loader,device=device)
trainer.train(5)
trainer.validate()
