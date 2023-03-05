import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"): ##计算两个边界框的交并比
    ##boxes_preds和boxes_labels分别代表预测的边界框和真实边界框，两者的格式可以是中心点和宽高，或者左上角和右下角的坐标。
    ##中心点和宽高格式的box_format为"midpoint"，左上角和右下角格式的box_format为"corners"。
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2 ##box1_x1左上角坐标
        ##boxes_preds[..., 0:1]表示预测框中心点的x坐标,boxes_preds[..., 2:3]表示预测框的宽
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2 ####box2_x1右下角坐标
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]  # (N, 1)
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)  ##y轴越往下越大
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)  ##clamp(0)的作用是将所有负数的值都变为0

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)  ##分母中可能会出现为0的情况


def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):  ##非极大值抑制
    ##bboxes 是一个包含多个预测框（bounding box）的列表，格式是(class_id, confidence, x1, y1, x2, y2)
    ##参数 iou_threshold 是 IOU 阈值，参数 threshold 是置信度阈值
    assert type(bboxes) == list ##这行代码是用来检查参数bboxes是否为列表类型的

    bboxes = [box for box in bboxes if box[1] > threshold] ##将原列表中所有满足 box[1] > threshold 条件的元素加入到新的列表 bboxes 中 b[1]是得分
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True) ##得分排序，lambda x: x[1] 表示使用第二个元素（即置信度）作为排序的关键字，reverse=True表从大到小
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]  ##这个框不与当前已选框 chosen_box 为同一个物体 该if语句为保留
               or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
               < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def mean_average_precision(  ##计算平均精度
        pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20
        ##预测框列表 pred_boxes、真实框列表 true_boxes、IOU 阈值 iou_threshold、框格式 box_format 和类别数 num_classes。
):
    ##存储每个类别的平均精度
    average_precisions = []

    ##用于数值稳定性
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []  ##真实目标框

        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        amount_bboxes = Counter([gt[0] for gt in ground_truths])
        ##计算每张图片中真实框的数量，从真实框列表ground_truths中提取每个真实框的图片索引gt[0]，并将这些索引统计到一个Counter对象中
        ##若 ground_truths 为 [[0, 0.1, 0.2, 0.3, 0.4], [0, 0.5, 0.6, 0.7, 0.8], [1, 0.2, 0.3, 0.4, 0.5]]，那么 amount_bboxes 将会是 {0: 2, 1: 1}，表示图像 0 出现了 2 次，图像 1 出现了 1 次

        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)  ##创建一个长度为该类别出现次数的全零张量

        detections.sort(key=lambda x: x[2], reverse=True) ##对置信度进行排序
        TP = torch.zeros((len(detections))) ##检测出的正样本数
        FP = torch.zeros((len(detections))) ##检测出的负样本数
        total_true_bboxes = len(ground_truths)  ##计算当前类别下的所有真实框的数量

        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]  ##ground_truth_img 列表筛选与当前检测框在同一图像中的真实框。

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union( ##计算IOU
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0: ##判断当前检测结果是否已经被标记过的
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            else:
                FP[detection_idx] = 1 ##如果IOU低则判断为假阳性

        TP_cumsum = torch.cumsum(TP, dim=0)  ##通过对它们进行累加操作，可以方便地得到召回率和精度的变化情况，即PR曲线
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon) ##召回率
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))  ##precision = TP / (TP + FP + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions)) ##在 precisions 的第一个位置插入一个 1，为了让 precisions 和 recalls 的长度一致，可以理解为前面累加过了这里得重新减回去

        recalls = torch.cat((torch.tensor([0]), recalls)) ## recall 数组的开头添加一个值为 0 的元素，原因是在计算 average precision 时需要从 (0,0) 开始计算。
        average_precisions.append(torch.trapz(precisions, recalls))
        ##torch.trapz()函数用于计算曲线下的积分面积，也就是计算精度-召回率曲线下的面积

    return sum(average_precisions) / len(average_precisions)


def plot_image(image, boxes):  ##在图片上绘制预测的边界框
    im = np.array(image)
    height, width, _ = im.shape ##第三个参数是通道数

    fig, ax = plt.subplots(1) ##fig是整个图像，ax是一个包含在fig中的子图，用于展示图像和矩形框  subplots(1)可以创建一个包含1个子图的Figure对象
    ax.imshow(im)
    ##在ax对象上显示图像的方式是使用imshow()方法，该方法将im数组解释为图像像素值，并在ax对象上显示出来。

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # 创建边框
    for box in boxes:
        box = box[2:] ##将框的左上角的坐标(x,y)去掉了，只保留了框的宽度和高度(w,h)。
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        ##如果assert语句的条件不满足，那么程序就会抛出一个AssertionError，并将双引号中的内容作为错误信息显示出来。
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height), ##矩形左上角的坐标，乘以图像的宽度和高度是因为这里使用的是归一化坐标；
            box[2] * width, ##box[2] * width：矩形的宽度，同样需要乘以图像的宽度以进行反归一化；
            box[3] * height, ##box[3] * height：矩形的高度，同样需要乘以图像的高度以进行反归一化；
            linewidth=1, ##绘制矩形边框的线宽；
            edgecolor="r", ##矩形边框的颜色；
            facecolor="none", ##矩形填充的颜色，这里设置为 "none" 表示不填充
        )
        ax.add_patch(rect)

    plt.show()


def get_bboxes(  ##获取一个数据集上所有预测框和真实框的列表
        loader, ##DataLoader 对象，用于加载数据集
        model, ##已经训练好的目标检测模型
        iou_threshold,  ##IoU 阈值，用于在进行非极大值抑制时判断两个框是否重叠
        threshold,  ##阈值，用于过滤预测框中置信度小于该阈值的框
        pred_format="cells",  ##预测框的格式
        box_format="midpoint",  ##预测框的坐标是相对于图像中心点的
        device="cuda",
):
    all_pred_boxes = []  ##预测框
    all_true_boxes = []  ##真实框

    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0

    for batch_idx, (x, labels) in enumerate(loader):
        ##x 是包含图像数据的张量；labels 是包含每张图像对应的真实框的列表
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():  ##在进行模型推理时，关闭 PyTorch 的自动求导机制，以减少内存的使用并加速代码的执行
            predictions = model(x)

        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels) ##将边界框从单元格坐标转换为绝对坐标的函数
        bboxes = cellboxes_to_boxes(predictions)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box) ##可以跟踪每个预测框是从哪个样本中预测出来的。

            for box in true_bboxes[idx]:
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes


def convert_cellboxes(predictions, S=7):  ##将模型预测的yolo格式的bounding box转换为真实坐标的函数
    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, 7, 7, 30)
    bboxes1 = predictions[..., 21:25]
    ##(batch_size, S, S, 4)
    bboxes2 = predictions[..., 26:30]
    scores = torch.cat(
        (predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)), dim=0  ##predictions[..., 20]每个值都是置信度
        ##形成一个形状为 (2, batch_size, S, S) 的新张量 scores
        ##原因是unsqueeze(0) 在第 0 维度（即最外层的维度）上增加了一维，使得原来的 (batch_size, 7, 7, 30) 的张量变成了 (1, batch_size, 7, 7, 30),
        ##而predictions[..., 20]又解开了一维因此predictions[..., 20].unsqueeze(0)变成了(1, batch_size, 7, 7),拼接后就是(2, batch_size, 7, 7)
    )
    best_box = scores.argmax(0).unsqueeze(-1) ##(batch_size, 7, 7, 1)的tensor，其中的最后一个维度是unsqueeze后添加的，argmax返回的是下标。
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    ##best_box的值是一个索引，表示在哪一个 box（bboxes1 或者 bboxes2）中的置信度最高。所以这里 best_box 只有两个取值 0 或者 1
    ##如果best_box为0，那么就选取bboxes1的预测结果；如果best_box为1，就选取bboxes2的预测结果
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)  ##表示每个单元格在网格中的位置 (batch_size, S, S, 1)
    ##cell_indices 中的横坐标和纵坐标分别处于不同的维度中，我们需要将其进行转置后才能进行相加。
    x = 1 / S * (best_boxes[..., :1] + cell_indices)  ## x 是边界框左上角 x 坐标相对于格子左上角 x 坐标的偏移量
    ##通俗来说就是每个网格都有对应的两个预测框，上面best_boxes表示每个网格对应的最佳预测框的x,y坐标，把他们加到每个网格上，再乘1/S即得到了偏移量
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / S * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_y), dim=-1) ##(batch_size, S, S, 3)
    predicted_class = predictions[..., :20].argmax(-1).unsqueeze(-1) ##类别下标
    best_confidence = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(-1)
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )

    return converted_preds ##表示每个网格点对应的预测框的坐标信息和物体类别概率。


def cellboxes_to_boxes(out, S=7): ##转为列表
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, -1)
    ##(batch_size, S, S, num_anchors, 5 + num_classes) -->(batch_size, S * S * num_anchors, 5 + num_classes)
    converted_pred[..., 0] = converted_pred[..., 0].long() ##目的是之后计算损失
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(S * S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
