import cv2
import numpy as np
import torch
from tqdm import tqdm
import os
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# img = cv2.imread('./data/images/2022-12-01_00-04-43_C20.jpg')
# img = cv2.resize(img, (640, 640))
# img_tofeed = img / 255.0
# x = torch.from_numpy(img_tofeed).float().unsqueeze(0).to(device)
# x = x.transpose(1, 3)
# print(x.size())
# height, width = img_tofeed.shape[:2]
# cell_height = height // 20
# cell_width = width // 20 
# model = torch.load('./models/exp/20230617.pt')
# outputs = model(x)
# output = outputs.cpu().detach().numpy()
# for index in range(len(outputs)):
#     for h_index in range(0,20):
#         for w_index in range(0,20):
#             # print(outputs[index,:,h_index,w_index].unsqueeze(0))
#             _, predicted = torch.max(outputs[index,:,h_index,w_index].unsqueeze(0), 1)
#             print(outputs[index,:,h_index,w_index][1])
#             print(predicted)
#             if predicted == 1:
#                 confidence = round(float(outputs[index,:,h_index,w_index][1]),4)
#                 start_x, end_x = w_index*cell_width, (w_index+1)*cell_width  
#                 start_y, end_y = h_index*cell_height, (h_index+1)*cell_height   
#                 cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (0,0,255), 2)
#                 text_x = start_x + cell_width // 2 - 10
#                 text_y = start_y + cell_height // 2 + 5
#                 cv2.putText(img, str(confidence), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,  
#                         0.6, (252,223,3), 2)  
# cv2.imwrite('result.jpg', img)            
def detect_and_draw(img_folder, save_folder,model):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    for img_name in tqdm(os.listdir(img_folder)):
        img = cv2.imread(os.path.join(img_folder, img_name))
        img = cv2.resize(img, (640, 640))
        img_tofeed = img / 255.0
        # 转换为 PyTorch 张量并放到 GPU 上
        x = torch.from_numpy(img_tofeed).float().unsqueeze(0).to(device)
        x = x.transpose(1, 3)
        height, width = img_tofeed.shape[:2]
        cell_height = height // 20
        cell_width = width // 20
        outputs = model(x)
        print(outputs)
        output = outputs.cpu().detach().numpy()
        for index in range(len(outputs)):
            for h_index in range(0,20):
                for w_index in range(0,20):
                    _, predicted = torch.max(outputs[index,:,h_index,w_index].unsqueeze(0), 1)
                    if predicted == 1:
                        confidence = round(float(outputs[index,:,h_index,w_index][1]),4)
                        start_x, end_x = w_index*cell_width, (w_index+1)*cell_width
                        start_y, end_y = h_index*cell_height, (h_index+1)*cell_height
                        cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (0,0,255), 2)
                        text_x = start_x + cell_width // 2 - 10
                        text_y = start_y + cell_height // 2 + 5
                        cv2.putText(img, str(confidence), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (252,223,3), 2)
        save_path = os.path.join(save_folder, img_name) 
        print(save_path)
        cv2.imwrite(save_path, img)
if __name__ == "__main__":
    model = torch.load('./models/exp/20230617.pt')
    os.system("./models/exp/pred_result/*")
    detect_and_draw("./data/images/","./models/exp/pred_result/",model)
