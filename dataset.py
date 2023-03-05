import torch
import os
import pandas as pd
from PIL import Image


class VOCDataset(torch.utils.data.Dataset):
    def __init__(
            self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None,
    ):
        self.annotations = pd.read_csv(csv_file) ##标注文件的路径
        self.img_dir = img_dir  ##图像文件的目录
        self.label_dir = label_dir  ##标签文件的目录
        self.transform = transform  ##数据增强和预处理函数
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        ##获取第index行、第1列（索引从0开始）的元素，即获取annotations数据集中第index个样本的标签文件名
        ##将 self.label_dir（标签文件所在目录）和 self.annotations.iloc[index, 1]（第index个样本对应的标签文件名）连接起来，得到该样本的标签文件路径
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()  ##将标签字符串中的每个数值分开
                ]

                boxes.append([class_label, x, y, width, height])

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)

        if self.transform:
            image, boxes = self.transform(image, boxes)

        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            i, j = int(self.S * y), int(self.S * x)  ##算出在哪一个网格中
            x_cell, y_cell = self.S * x - j, self.S * y - i ##偏移量

            width_cell, height_cell = ( ##表示当前框相对于所在的 grid cell 的宽度和高度
                width * self.S,  ##从图像坐标系转化到 grid cell 的坐标系
                height * self.S,
            )

            if label_matrix[i, j, 20] == 0:  ##避免同一个位置上有多个bounding box
                label_matrix[i, j, 20] = 1

                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )

                label_matrix[i, j, 21:25] = box_coordinates

                label_matrix[i, j, class_label] = 1

        return image, label_matrix
