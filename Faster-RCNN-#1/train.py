import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from WheatData import train_data_loader, valid_data_loader

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
    #optimizer = torch.optim.Adam(params, lr=1e-3)
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = None

    loss_hist = Averager()
    val_loss_hist = Averager()
    itr = 1
    least_loss = float('inf')

    for epo in range(epoch):
        loss_hist.reset()
        val_loss_hist.reset()
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
                print(f"Iteration #{itr} loss: {loss_value}")

            itr += 1

        # 验证
        model.eval()
        with torch.no_grad():
            for val_images, val_targets, val_image_ids in valid_data_loader:
                images = list(image.to(device) for image in val_images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in val_targets]

                val_loss_dict = model(images, targets)  # Returns losses and detections
                val_losses = sum(loss for loss in val_loss_dict.values())
                loss_value = losses.item()
                loss_hist.send(loss_value)

        # 判断是否为更优的模型，以loss为标准
        if val_loss_hist.value<least_loss:
            least_loss = val_loss_hist.value
            lval=int(least_loss*1000)/1000
            torch.save(model.state_dict(), f'fasterrcnn_custom_test_ep{epo}_loss{lval}.pth')
            
        else:
            if lr_scheduler is not None:
                lr_scheduler.step()
        print(f"Epoch #{epo} train_loss: {loss_hist.value} val_loss: {val_loss_hist.value}")
