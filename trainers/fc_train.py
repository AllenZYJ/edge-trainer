import torch
from trainers.trainer import Trainer

class FCTrainer(Trainer):
    def train_epoch(self):
        self.model.train()
        for x, y in self.train_loader:
            outputs = self.model(x)
            loss = self.loss_fn(outputs, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    def validate(self):
        self.model.eval()
        val_loss = 0.
        val_acc = 0.
        with torch.no_grad():
            for x, y in self.val_loader:
                outputs = self.model(x)
                loss = self.loss_fn(outputs, y)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                val_acc += (predicted == y).sum().item()
                
        val_loss /= len(self.val_loader)
        val_acc = val_acc / len(self.val_loader.dataset) 
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')