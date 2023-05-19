import cv2
import numpy as np
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
img = cv2.imread('./data/images/2022-12-01_00-04-43_C20.jpg')
img = cv2.resize(img, (640, 640))
img = img / 255.0
# 转换为 PyTorch 张量并放到 GPU 上 
x = torch.from_numpy(img).float().unsqueeze(0).to(device)
x = x.transpose(1, 3)  
print(x.size())
model = torch.load('./models/exp/last.pt')
outputs = model(x)
output = outputs.cpu().detach().numpy()
print(output[0][1])
print(output[0][0])