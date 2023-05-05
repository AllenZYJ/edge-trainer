import torch 
from torch import nn
from torch.utils.data import DataLoader

from models.fc_model import FCModel 
from trainers.fc_train import FCTrainer

# 生成模拟数据
input_size = 64
hidden_size = 32
output_size = 10

x = torch.rand(100, input_size) 
y = torch.randint(0, output_size, (100,))

# 定义数据集
dataset = torch.utils.data.TensorDataset(x, y)

# 定义数据加载器 
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(dataset, batch_size=16) 
# 模型,优化器和损失函数 
model = FCModel(input_size, hidden_size, output_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()
# 训练模型
trainer = FCTrainer(model, optimizer, loss_fn, train_loader, val_loader)
trainer.train(5) 
trainer.validate()