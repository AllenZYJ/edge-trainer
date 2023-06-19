import torch

class Trainer:
    def __init__(self, model, optimizer, loss_fn, train_loader, val_loader,device):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
    def train(self, epochs):
        for epoch in range(epochs):
            print(f'Epochs: {epoch}') 
            self.train_epoch()
            # self.validate()
            
    def train_epoch(self):
        self.model.train()
        for x, y in self.train_loader:
            # 训练逻辑
            ... 
            
    def validate(self):
        self.model.eval()
        with torch.no_grad():
            for x, y in self.val_loader:
                pass