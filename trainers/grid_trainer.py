import torch
from torch import nn
from trainers.trainer import Trainer
from log.edge_log import Logger
from tqdm import tqdm
import time
logger = Logger('logs/')
log_name = "training"
logger.create_log_file(log_name)
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
            loss = 0.0
            count_a_sample=0
            for index in range(len(outputs)):
                for h_index in range(0,self.model.grid_shape):
                    for w_index in range(0,self.model.grid_shape):
                        loss += self.loss_fn(outputs[index,:,h_index,w_index].unsqueeze(0), y[index,:,h_index,w_index]) 
                        _, predicted = torch.max(outputs[index,:,h_index,w_index].unsqueeze(0), 1)
                        if y[index,:,h_index,w_index] == predicted:
                            count_a_sample+=1
            loss/=(self.model.grid_shape*self.model.grid_shape*len(y))
            logger.write_log(log_name, f'Train Loss: {loss:.4f}')
            one_train_acc += count_a_sample / (self.model.grid_shape*self.model.grid_shape*len(y))*100# 单个样本准确率
            loss.backward() 
            self.optimizer.step()
            self.optimizer.zero_grad()
        # print("one_train_acc:",one_train_acc/10)
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
                    for h_index in range(0,self.model.grid_shape):
                        for w_index in range(0,self.model.grid_shape):
                            loss += self.loss_fn(outputs[index,:,h_index,w_index].unsqueeze(0), y[index,:,h_index,w_index]) 
                            _, predicted = torch.max(outputs[index,:,h_index,w_index].unsqueeze(0), 1)
                            if y[index,:,h_index,w_index] == predicted:
                                count_a_sample+=1
                val_loss += loss
            #     # Backpropagation
                one_acc_batch = count_a_sample/(self.model.grid_shape*self.model.grid_shape*len(y))
                one_val_acc += one_acc_batch
        val_acc = one_val_acc/len(self.val_loader)*100
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')