import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from WheatData import train_data_loader, valid_data_loader
from evaluate import *

class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0

def train(epoch=200):
    # 准备网络
    # load a model; pre-trained on COCO（以下4句为pytorch官方教程例子）
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 2  # 1 class (wheat) + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 指定gpu
    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')


    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=1e-3)
    #optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = None

    loss_hist = Averager()
    valid_pred_min = 0.5
    #val_loss_hist = Averager()
    #least_loss = float('inf')

    for epo in range(epoch):
        itr = 1
        loss_hist.reset()
        #val_loss_hist.reset()
        model.train()
        for images, targets, image_ids in train_data_loader:
            
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            # 这里应该是model中封装了求loss，官方教程中写的是 model(images, targets):Returns losses and detections
            loss_dict = model(images, targets)  # Returns losses and detections
            # 对一个batch的loss求和
            losses = sum(loss for loss in loss_dict.values())
            # item方法是取出数值
            loss_value = losses.item()
            # 使用封装的类记录loss
            loss_hist.send(loss_value)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if itr % 50 == 0:
                line_1 = f"Iteration #{itr} loss: {loss_value}"
                print(f"Iteration #{itr} loss: {loss_value}")
                with open('log/logs_frc_0629.txt', 'a') as f :
                    f.write(line_1)
                    f.write('\r\n')
            itr += 1

        # 验证
        model.eval()
        validation_image_precisions = []
        iou_thresholds = [x for x in np.arange(0.5, 0.76, 0.05)]
        with torch.no_grad():
            for val_images, val_targets, val_image_ids in valid_data_loader:
                images = list(image.to(device) for image in val_images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in val_targets]

                outputs = model(images)
                for i, image in enumerate(images):
                    boxes = outputs[i]['boxes'].data.cpu().numpy()
                    scores = outputs[i]['scores'].data.cpu().numpy()
                    gt_boxes = targets[i]['boxes'].cpu().numpy()
                    # np.argsort(scores)返回scores从小到大的索引；[::-1]表示倒序，从大到小
                    preds_sorted_idx = np.argsort(scores)[::-1]
                    preds_sorted = boxes[preds_sorted_idx]
                    image_precision = calculate_image_precision(preds_sorted,
                                                        gt_boxes,
                                                        thresholds=iou_thresholds,
                                                        form='pascal_voc')
                    validation_image_precisions.append(image_precision)
        valid_prec = np.mean(validation_image_precisions)
        line_2 = "Validation IOU: {0:.4f}".format(valid_prec)
        print("Validation IOU: {0:.4f}".format(valid_prec))
        with open('log/logs_frc_0629.txt', 'a') as f :
            f.write(line_2)
            f.write('\r\n')

        # 保存更优的模型
        if valid_prec >= valid_pred_min:
            line_3 = 'Validation precision increased({:.6f} --> {:.6f}).  Saving model ...'.format(valid_pred_min, valid_prec)
            print('Validation precision increased({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_pred_min,
            valid_prec))
            with open('log/logs_frc_0629.txt', 'a') as f :
                f.write(line_3)
                f.write('\r\n')
            torch.save(model.state_dict(), 'models_frc_0629/frc_0629_{}.pth'.format(epo))
            valid_pred_min = valid_prec
        line_4 = f"Epoch #{epo} train_loss: {loss_hist.value}"
        print(f"Epoch #{epo} train_loss: {loss_hist.value}")
        with open('log/logs_frc_0629.txt', 'a') as f :
            f.write(line_4)
            f.write('\r\n')

if __name__ == "__main__":
    train()
