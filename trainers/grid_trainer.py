import torch
from torch import nn
from trainers.trainer import Trainer
class grid_trainer(Trainer):
    def train_epoch(self):  
        self.model.train()
        self.model.to(self.device)
        train_running_loss = 0.0
        train_running_correct = 0
        for x, y in self.train_loader:
            x=x.to(self.device)
            y=y.to(self.device)
            outputs = self.model(x)
            print(outputs.shape)
            print(y.shape)
            loss = 0.0
            for i in range(0,self.model.grid_shape,2):
                loss += self.loss_fn(outputs[:,i:i+2], y[:,i//2]) 
            train_running_loss += loss.item()
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            train_running_correct += (preds == y).sum().item()
            # Backpropagation
            loss.backward()
            # Update the weights.
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
                loss = 0.0
                for i in range(0,self.model.grid_shape,2):
                    loss += self.loss_fn(outputs[:,i:i+2], y[:,i//2]) 
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_acc += (predicted == y).sum().item()
        val_loss /= len(self.val_loader)
        val_acc = val_acc / len(self.val_loader.dataset)
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')