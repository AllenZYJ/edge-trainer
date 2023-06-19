import torch
from torch import nn
from torch.utils.data import DataLoader
from models.grid_net import ResNet,BasicBlock
from datasets.unname_dataset import GridDataset
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random
import time
import numpy as np
from log.edge_log import Logger
from tqdm import tqdm
import pickle5

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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
batchsize = 32
usenocache = False
save_flash = False
def main():
    print(batchsize)
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = GridDataset('./data/images/','./data/labels/', transform)
    x = [] 
    if usenocache:
        pbar = tqdm(range(len(dataset)))
        pbar.set_description('Scaning')
        for i in pbar:   
            x.append(dataset[i][0])
        x = torch.stack(x, dim=0)
        if save_flash:
            with open('./data/x_all.pkl', 'wb') as f:
                pickle5.dump(x, f)
        y=[]
        for image, label in dataset: 
            y.append(label.unsqueeze(0))
        y = torch.cat(y, dim=0).unsqueeze(1)
        if save_flash:
            with open('./data/y_all.pkl', 'wb') as f:
                pickle5.dump(y, f)
    else:
        print("load dataset from cache...")
        with open('./data/x_all.pkl', 'rb') as f:
            x = pickle5.load(f)
        with open('./data/y_all.pkl', 'rb') as f:
            y = pickle5.load(f)
    print("x:",type(x)) 
    print("y:",y.size())
    print("x:",x.size()) # torch.Size([N, 3, 640, 640])
    # 定义数据集
    dataset = torch.utils.data.TensorDataset(x, y)
    # 定义数据加载器
    train_loader = DataLoader(dataset, batch_size=batchsize, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=batchsize)
    # 模型,优化器和损失函数
    model = ResNet(img_channels=3, num_layers=18, block=BasicBlock, grid_shape=20)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    # 训练模型 
    from trainers.grid_trainer import grid_trainer
    trainer = grid_trainer(model, optimizer, loss_fn, train_loader, val_loader,device=device)
    trainer.train(10)
    trainer.validate()
    torch.save(model, './models/exp/20230618.pt')


if __name__ == '__main__':
    main()