import os 
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import cv2
class GridDataset(Dataset):
    def __init__(self, img_dir, label_dir,transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = os.listdir(img_dir)[:10]
        self.labels = {img_name: img_name[:-3]+'txt'
                      for img_name in self.images}
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        label_path = os.path.join(self.label_dir, self.labels[self.images[idx]])
        label = open(label_path, 'r')
        label = label.read().split()
        label = torch.tensor(list(map(int, label)))
        label = label.view(20, 20)
        print(img_path)
        image = cv2.imread(img_path)
        image = cv2.resize(image,(640, 640))/255.0
        if self.transform:
            image = self.transform(image)
        return image.float(),label