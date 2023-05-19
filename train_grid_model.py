import torch
from torch import nn
from torch.utils.data import DataLoader
from models.grid_net import ResNet,BasicBlock
from data.unname_dataset import GridDataset
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random
import time
import numpy as np
from log.edge_log import Logger
from tqdm import tqdm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
def main():
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = GridDataset('./data/images/','./data/labels/', transform)
    x = [] 
    pbar = tqdm(range(len(dataset)))
    pbar.set_description('Scaning')
    for i in pbar:   
        x.append(dataset[i][0])
    x = torch.stack(x, dim=0)
    print("x:",x.size()) # torch.Size([N, 3, 640, 640])
    y=[]
    for image, label in dataset: 
        y.append(label.unsqueeze(0))
    y = torch.cat(y, dim=0).unsqueeze(1)
    print("y:",y.size()) # torch.Size([N, 1, 20, 20])
    # x = torch.rand(1000, 3, input_size, input_size)  
    # y = torch.randint(0,2,(1000,1,20,20))
    # 定义数据集
    dataset = torch.utils.data.TensorDataset(x, y)
    # 定义数据加载器
    train_loader = DataLoader(dataset, batch_size=48, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=48)
    # 模型,优化器和损失函数
    # model = gridnet(batchsize=1)
    model = ResNet(img_channels=3, num_layers=18, block=BasicBlock, grid_shape=20)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    # 训练模型 
    from trainers.grid_trainer import grid_trainer
    trainer = grid_trainer(model, optimizer, loss_fn, train_loader, val_loader,device=device)
    trainer.train(50)
    trainer.validate()
    torch.save(model, './models/exp/2023-05-17-last.pt')
    # model = torch.load('./models/exp/last.pt')
    # for x, y in train_loader:
    #     x=x.to(device)
    #     y=y.to(device)
    #     outputs = model(x)
    #     loss = 0.0
    #     count_a_sample=0
    #     print(outputs.shape)

if __name__ == '__main__':
    main()