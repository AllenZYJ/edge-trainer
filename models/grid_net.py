from torch import nn
import torch.nn.functional as F
class gridnet(nn.Module):
    def __init__(self,hidden_size, output_size):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, (7, 7))  
        self.pool = nn.MaxPool2d((2, 2), 2)   

        self.conv2 = nn.Conv2d(6, 6, 5) 
        self.fc1 = nn.Linear(6 * 156 * 156, hidden_size)   
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        print(x.shape)
        x = x.view(-1, 6 * 156 * 156) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = nn.Softmax(dim=1)(x)  
        return x
