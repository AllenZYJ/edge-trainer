import torch
from torch import nn
from trainers.trainer import Trainer
class grid_trainer(Trainer):
    def train_epoch(self):  
        self.model.train()
        self.model.to(self.device)
        for x, y in self.train_loader:
            x=x.to(self.device)
            print("x:",x.shape)
            y=y.to(self.device)
            print("y:",y.shape)
            outputs = self.model(x)
            loss = self.loss_fn(outputs, y)
            print(loss)
            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),5)  
            print(f'Gradient norm: {grad_norm}')
            self.optimizer.step()
    def validate(self):
        self.model.eval()
        val_loss = 0.
        val_acc = 0.
        with torch.no_grad():
            for x, y in self.val_loader:
                x=x.to(self.device)
                y=y.to(self.device)
                outputs = self.model(x)
                loss = self.loss_fn(outputs, y)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_acc += (predicted == y).sum().item()
        val_loss /= len(self.val_loader)
        val_acc = val_acc / len(self.val_loader.dataset)
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')