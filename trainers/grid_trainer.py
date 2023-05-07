import torch
from torch import nn
from trainers.trainer import Trainer
from tqdm import tqdm
class grid_trainer(Trainer):
    def train_epoch(self):  
        self.model.train()
        self.model.to(self.device)
        train_running_loss = 0.0
        train_running_correct = 0
        pbar = tqdm(self.train_loader)
        pbar.set_description('Training')
        one_train_acc = 0.
        for x, y in pbar:
            x=x.to(self.device)
            y=y.to(self.device)
            outputs = self.model(x)
            print(outputs.shape)
            loss = 0.0
            count_a_sample=0
            for index in range(len(outputs)):
                for i in range(0,self.model.grid_shape,2):
                    loss += self.loss_fn(outputs[index,0:2].unsqueeze(0), y[index, i//2].unsqueeze(0)) 
                    _, predicted = torch.max(outputs[index,0:2].unsqueeze(0), 1)
                    if y[index,i//2] == predicted:
                        count_a_sample+=1
                    one_train_acc += count_a_sample / len(y[0])*100
                train_running_loss += loss.item()
                _, preds = torch.max(outputs.data, 1)
            # Backpropagation
            loss.backward()
            # Update the weights.
            self.optimizer.step()
        # tensor([[0.1433, 0.0000]], device='cuda:0', grad_fn=<SliceBackward>)
        # tensor([1], device='cuda:0')


        # for x, y in pbar:
        #     x=x.to(self.device)
        #     y=y.to(self.device)
        #     outputs = self.model(x)
        #     loss = 0.0
        #     for i in range(0,self.model.grid_shape,2):
        #         print(outputs[:,i:i+2], y[:,i//2])
        #         loss += self.loss_fn(outputs[:,i:i+2], y[:,i//2]) 
        #     train_running_loss += loss.item()
        #     # Calculate the accuracy.
        #     _, preds = torch.max(outputs.data, 1)
        #     train_running_correct += (preds == y).sum().item()
        #     # Backpropagation
        #     loss.backward()
        #     # Update the weights.
        #     self.optimizer.step()
        print("loss:",loss)
    def validate(self):
        self.model.eval()
        val_loss = 0.
        val_acc = 0.
        one_val_acc = 0.
        with torch.no_grad():
            for x, y in self.val_loader:
                x=x.to(self.device)
                y=y.to(self.device)
                outputs = self.model(x)
                loss = 0.0
                count_a_sample=0
                for index in range(len(outputs)):
                    for i in range(0,self.model.grid_shape,2):
                        loss += self.loss_fn(outputs[index,0:2].unsqueeze(0), y[index, i//2].unsqueeze(0)) 
                        _, predicted = torch.max(outputs[index,0:2].unsqueeze(0), 1)
                        if y[index,i//2] == predicted:
                            count_a_sample+=1
                    one_val_acc += count_a_sample / len(y[0])*100
            val_acc += one_val_acc/len(self.val_loader)/len(outputs)
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')