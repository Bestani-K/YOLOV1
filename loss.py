import torch
import torch.nn as nn
from utils import intersection_over_union


class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")    ##均方误差（MSE）损失函数  reduction 参数为 "sum"，表示将所有样本的损失求和得到最终的损失值。
        ##MSE Loss = (1 / n) * ∑(i=1 -> n)(y_pred_i - y_true_i) ^ 2
        self.S = S  #S即特征图的大小；
        self.B = B  #B即每个特征点要预测的边界框数；
        self.C = C  #C即要预测的目标类别数
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B*5) ##-1自动推断大小

        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25]) ##(batch_size, S, S, B)
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        ##计算每个bbox与target的iou
        iou_maxes, bestbox = torch.max(ious, dim=0)  ##ios_maxes是值，bestbox是索引
        exists_box = target[..., 20].unsqueeze(3) ##(batch_size,S,S) --> (batch_size, S, S, 1)
        ##target的第21的值是该边界框是否包含物体的二元标志（0表示没有物体，1表示有物体）；
        ##接下来的两个值是边界框的中心点坐标在该网格单元格中的偏移量（偏移值范围为0到1，相对于该网格单元格的宽度和高度）；
        ##然后是边界框的宽度和高度，也是相对于整个图像的比例（即值在0到1之间）
        ####!一个格子内最多只有一个真实框，所以如果存在第二个真实框的话，它们的信息就保存在了相邻的格子中。而在当前这个格子内，第26-30个值记录的是该格子内第一个真实框的信息。!

        box_predictions = exists_box * (  ##exists_box表是否存在预测框，bestbox记录哪个框好，predictions记录坐标信息
            (
                    bestbox * predictions[..., 26:30]
                    + (1 - bestbox) * predictions[..., 21:25]
            )
        )

        box_targets = exists_box * target[..., 21:25]  ##真实框的信息

        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            ##对预测框的长宽进行调整(即w, h)，torch.sign 用于获取每个元素的正负号，即如果是正数则为1，如果是负数则为-1
            torch.sqrt(box_predictions[..., 2:4] + 1e-6)
        )

        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
            ##x与y已经包括在其中了
            ##box_predictions的形状是(batch_size, grid_size, grid_size, num_classes + 5)展开后是(batch_size, grid_size * grid_size * (num_classes + 5))
        )

        # 负责检测物体的bbox的 confidence误差
        pred_box = (
                bestbox * predictions[..., 25:26] + (1 - bestbox) * predictions[..., 20:21]
        )

        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21]),
        )

        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1), ##--> (batch_size * grid_size * grid_size * num_boxes)
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )

        # 类别预测误差
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2, ),
            torch.flatten(exists_box * target[..., :20], end_dim=-2, ),
        )

        loss = (
                self.lambda_coord * box_loss  # first two rows in paper
                + object_loss  # third row in paper
                + self.lambda_noobj * no_object_loss  # forth row
                + class_loss  # fifth row
        )

        return loss

