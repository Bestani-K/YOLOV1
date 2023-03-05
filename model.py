import torch
import torch.nn as nn

architecture_config = [
##第一个参数卷积核大小7*7，64个过滤器(即通道数)，2步长，3填充 !!padding的大小一般为卷积核的一半!!
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]
##该数据是yolo卷积层结构


class CNNblock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        ##bias是指卷积层中的偏置项，即每个卷积核都有一个偏置项，可以用来调整输出的平均值。偏置项可以帮助模型更好地适应数据集
        super(CNNblock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        ##归一化层，有助于加速模型的训练，防止梯度消失或爆炸，并提高模型的泛化能力。
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))

class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        ##darknet是一个由一系列卷积层组成的神经网络，它是 YOLOv1 模型的骨干网络
        self.fcs = self._create_fcs(**kwargs)
        ##self.fcs 是用来做后续物体检测的全连接层，用于将卷积层提取出来的特征映射转化为具体的物体类别和位置信息。

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))
        ##torch.flatten(x, start_dim=1)将x沿着start_dim=1开始的维度展平为一个一维张量，并作为全连接层的输入。全连接层的输出即为模型的最终输出。

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple: ##判断x是否是元组类型的
                layers += [
                    CNNblock(
                    in_channels, out_channels=x[1], kernel_size=x[0], stride=x[2], padding=x[3]
                    )
                ]

                in_channels = x[1]

            elif type(x) == str: ##即‘M'
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

            elif type(x) == list: #类似[(1, 256, 1, 0), (3, 512, 1, 1), 4],
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]

                for _ in range(num_repeats):
                    layers += [
                        CNNblock(
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3]
                        )
                    ]

                    layers += [
                        CNNblock(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3]
                        )
                    ]

                    in_channels = conv2[1]

        return nn.Sequential(*layers)


    def _create_fcs(self, split_size, num_boxes, num_classes): ##后面的全连接层
        ##用于将Darknet网络的输出特征映射转换成检测框、类别以及置信度的预测值。
        S, B, C = split_size, num_boxes, num_classes
        ##split_size，即特征图的大小；num_boxes，即每个特征点要预测的边界框数；num_classes，即要预测的目标类别数
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 496), ##num_features=1024
            nn.Dropout(0.0),
            #是一个用于模型训练中的正则化技术，作用是防止过拟合，它会随机将一定比例的输入元素设为0，其实现方式是对输入张量中的每个元素以概率p随机置为0，而对于被保留的元素，其值会被等比例放大以保证张量的期望值不变。
            nn.LeakyReLU(0.1),
            nn.Linear(496, S * S * (C + B * 5))
            ##每个边界框需要预测 5 个值：边界框的中心坐标 x 和 y，边界框的宽度和高度以及对象的置信度（confidence score）。
        )

# def test(split_size=7, num_boxes=2, num_classes=20):
#         model = Yolov1(split_size=split_size, num_boxes=num_boxes, num_classes=num_classes)
#         x = torch.randn((2, 3, 448, 448))
#         #第一个维度是batch size，为2；第二个维度是通道数，为3，表示RGB三个通道；第三个和第四个维度是图像的高和宽，为448像素
#         print(model(x).shape)

# test() torch.Size([2, 1470])
# 模型输出的张量应该是 (batch_size, split_size*split_size*num_boxes*(5+num_classes)) 的大小，其中 batch_size 是输入张量的批次数。对于您提供的代码中的输入张量， batch_size=2。因此，输出张量的大小为 (2, 1470)。