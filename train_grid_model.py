import torch
from torch import nn
from torch.utils.data import DataLoader
from models.grid_net import gridnet

# 生成模拟数据
input_size = 640
hidden_size = 32
output_size = 1080

x = torch.rand(100, 3, input_size, input_size)  
y = torch.randint(0, output_size, (100,))    

print(x.shape)
print(y.shape)
# 定义数据集
dataset = torch.utils.data.TensorDataset(x, y)

# 定义数据加载器
train_loader = DataLoader(dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(dataset, batch_size=2)

# 模型,优化器和损失函数
model = gridnet(hidden_size, output_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

# 训练模型 
from trainers.grid_trainer import grid_trainer
trainer = grid_trainer(model, optimizer, loss_fn, train_loader, val_loader)
trainer.train(5)
trainer.validate()
