import json 
import os
from tqdm import tqdm
import shutil
def generate_label(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    height = data['imageHeight']
    width = data['imageWidth']
    num_grid = 20 # 将图片分成20*20的格子

    labels = []
    for i in range(num_grid):
        for j in range(num_grid):
            labels.append(0)
    for shape in data['shapes']:
        # 获取shape的类型和坐标
        label = shape['label'] 
        points = shape['points']
        x1 = points[0][0]
        y1 = points[0][1]
        x2 = points[1][0]
        y2 = points[1][1]
        x_center = (x1 + x2) / 2 # 计算shape的中心点
        y_center = (y1 + y2) / 2
       
        # 计算中心点在第几个格子内
        grid_x = int(x_center / (width / num_grid))
        grid_y = int(y_center / (height / num_grid))
        labels[grid_y * num_grid + grid_x] = 1  # 将对应格子标记为1
    return labels
def generate_label_batch(label_path):
    label_files = [f for f in os.listdir(label_path) if f.endswith('.json')]
    labels_batch = []
    for filename in tqdm(label_files[:]):
        if filename.endswith('.json'):
            if os.path.exists(os.path.join(label_path,filename.replace('.json', '.jpg'))):
                img_file = filename.replace('.json', '.jpg')
                shutil.copy(os.path.join(label_path, img_file), "./data/images")
                labels = generate_label(os.path.join(label_path, filename))
            elif os.path.exists(os.path.join(label_path,filename.replace('.json', '.png'))):
                img_file = filename.replace('.json', '.png')
                shutil.copy(os.path.join(label_path, img_file), "./data/images")
                labels = generate_label(os.path.join(label_path, filename))
            elif os.path.exists(os.path.join(label_path,filename.replace('.json', '.jpeg'))):
                continue
                # img_file = filename.replace('.json', '.jpeg')
                # print("jpeg",img_file)
                # shutil.copy(os.path.join(label_path, img_file), "./data/images")
                # labels = generate_label(os.path.join(label_path, filename))
        with open(os.path.join("./data/labels", filename[:-4]+"txt"), 'w') as f:
            for label in labels:
                f.write(str(label) + ' ')
    return labels_batch
if __name__ == '__main__':
    # labels = generate_label('/backup/datasets/dataset-anticollision-newhd/anti-collision/images/vlc-record-2023-03-09-17h25m41s-rtsp___10.7.3.213-_00320.json')
    generate_label_batch("/backup/datasets/dataset-anticollision-newhd/anti-collision/images/")