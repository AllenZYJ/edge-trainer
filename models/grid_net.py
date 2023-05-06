from torch import nn
import torch.nn.functional as F
import torch
from torch import nn
from torchvision.models import resnet18
class gridnet(nn.Module):
    def __init__(self, batchsize=1):
        super().__init__()
        self.batchsize = batchsize
        # ResNet18作为backbone
        self.backbone = resnet18(pretrained=False)  
        modules = list(self.backbone.children())[:-3]
        resnet_backbone = nn.Sequential(*modules)
        self.backbone = resnet_backbone
        # 在maxpool4后接上1x1卷积得到attention score
        # 在avgpool后接上全连接层得到embedding
        self.classifier = nn.Linear(256*40*40,2)  
        # 固定resnet,仅更新新增层
        for param in self.backbone.parameters():
            param.requires_grad = False
    def forward(self, x):
        F_backbone = self.backbone(x)
        x = self.classifier(F_backbone.view(self.batchsize,-1))
        x = F.softmax(x,dim=0)
        print(x.shape)
        return x
