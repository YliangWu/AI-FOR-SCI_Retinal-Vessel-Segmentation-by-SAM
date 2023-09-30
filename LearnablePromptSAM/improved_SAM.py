#coding:utf-8
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as opt
from PIL import Image
import argparse
import numpy as np
from albumentations import Compose, Resize, Normalize, ColorJitter, HorizontalFlip, VerticalFlip
import glob
import os
import re
import matplotlib.pyplot as plt
import monai
import pandas as pd
from datetime import datetime
device = "cuda" if torch.cuda.is_available() else "cpu"
print("using device:"+("cuda" if torch.cuda.is_available() else "cpu"))
run_id = datetime.now().strftime("%Y%m%d-%H%M")
parser = argparse.ArgumentParser("Learnable prompt")
parser.add_argument("--train_image", type=str, default='./dataset/train/img',
                    help="path to the image that used to train the model")
parser.add_argument("--train_mask_path", type=str, default='./dataset/train/mask',
                    help="path to the mask file for training")
parser.add_argument("--test_image", type=str, default='./dataset/test/img',
                    help="path to the image that used to train the model")
parser.add_argument("--test_mask_path", type=str, default='./dataset/test/mask',
                    help="path to the mask file for training")
parser.add_argument("--epoch", type=int, default=10,
                    help="training epochs")
parser.add_argument("--checkpoint", type=str, default="./weights/sam_vit_b_01ec64.pth",
                    help="path to the checkpoint of sam")
parser.add_argument("--model_name", default="vit_b", type=str,
                    help="name of the sam model, default is vit_h",
                    choices=["default", "vit_b", "vit_l", "vit_h"])
parser.add_argument("--save_path", type=str, default="./ckpt_prompt/"+run_id+"/",
                    help="save the weights of the model")
parser.add_argument("--num_classes", type=int, default=2)
parser.add_argument("--mix_precision", action="store_true", default=False,
                    help="whether use mix precison training")
parser.add_argument("--lr", type=float, default=0.05, help="learning rate")
parser.add_argument("--optimizer", default='sgd',#"adamw",
                    type=str, help="optimizer used to train the model")
parser.add_argument("--weight_decay", default=5e-4, type=float, 
                    help="weight decay for the optimizer")
parser.add_argument("--momentum", default=0.9, type=float,
                    help="momentum for the sgd")
parser.add_argument("--batch_size", default=2, type=int)
parser.add_argument("--divide", default=True,
                    help="whether divide the mask")
parser.add_argument("--divide_value", type=int, default=255,
                    help="divided value")
parser.add_argument("--num_workers", "-j", type=int, default=1, 
                    help="divided value")
parser.add_argument("--device", default='cuda', type=str)
parser.add_argument("--model_type", default="sam", choices=["dino", "sam"], type=str,
                    help="backbone type")

parser.add_argument("--restore_model",type=str,help="train base on the old model or not?")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.device

from learnerable_seg import PromptSAM, PromptDiNo
from scheduler import PolyLRScheduler
from metrics.metric import Metric
def plot_metrics(iou_values):
    epochs = len(iou_values)
    x = np.arange(1, epochs + 1)
    plt.plot(x, iou_values, label='iou scores')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.title('Training Metrics')
    plt.legend()
    plt.show()

def calculate_accuracy(pred_data, target_data):
    accuracy=[]
    for pred, target in zip(pred_data, target_data):
        pred_binary = (pred == 255)  # 将预测结果转换为二进制（True/False）
        target_binary = (target == 255)  # 将目标结果转换为二进制（True/False）
       # 将PyTorch张量从CUDA设备移动到主机内存上
        pred_binary = pred_binary.cpu().numpy()
        target_binary = target_binary.cpu().numpy()
        # 计算预测为血管且正确的像素数
        correct_pixels = np.sum(np.logical_and(pred_binary, target_binary))

        # 计算预测为血管的像素数
        predicted_pixels = np.sum(pred_binary)
        accuracy.append(correct_pixels/predicted_pixels)
        #print("predicted_pixels",predicted_pixels)
        #print("correct_pixels",correct_pixels)
    return sum(accuracy)/len(accuracy)
def calculate_recall(pred_data,target_data):
    recall=[]
    for pred,target in zip(pred_data,target_data):
        pred_binary=(pred==255)
        target_binary = (target ==255)
       # 将PyTorch张量从CUDA设备移动到主机内存上
        pred_binary = pred_binary.cpu().numpy()
        target_binary = target_binary.cpu().numpy()
        # 计算预测为血管且正确的像素数
        correct_pixels = np.sum(np.logical_and(pred_binary,target_binary))
        # 真实血管的像素数之比
        true_pixels=np.sum(target_binary)
        #print("true_pixels",true_pixels)
        recall.append(correct_pixels/true_pixels)
    return sum(recall)/len(recall)
class SegDataset:
    def __init__(self, img_paths, mask_paths, 
                 mask_divide=False, divide_value=255,
                 pixel_mean=[0.5]*3, pixel_std=[0.5]*3,
                 img_size=1024) -> None:
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.length = len(img_paths)
        self.mask_divide = mask_divide
        self.divide_value = divide_value
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        self.img_size = img_size
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        mask_path = self.mask_paths[index]
        img = Image.open(img_path).convert("RGB")   #使用RGB图
        img = np.asarray(img)
        mask = Image.open(mask_path).convert("L") #使用灰度图
        mask = np.asarray(mask)
        if self.mask_divide:
            mask = mask // self.divide_value    #0-1化
        transform = Compose(
            [
                ColorJitter(),
                VerticalFlip(),
                HorizontalFlip(),
                Resize(self.img_size, self.img_size),
                Normalize(mean=self.pixel_mean, std=self.pixel_std)
            ]
        )
        aug_data = transform(image=img, mask=mask)
        x = aug_data["image"]
        target = aug_data["mask"]
        if img.ndim == 3:
            x = np.transpose(x, axes=[2, 0, 1])
        elif img.ndim == 2:
            x = np.expand_dims(x, axis=0)
        return torch.from_numpy(x), torch.from_numpy(target)

def main(args):

    train_img_path = args.train_image
    train_mask_path = args.train_mask_path
    test_img_path = args.test_image
    test_mask_path = args.test_mask_path
    epochs = args.epoch
    checkpoint = args.checkpoint
    model_name = args.model_name
    save_path = args.save_path
    optimizer = args.optimizer
    weight_decay = args.weight_decay
    lr = args.lr
    momentum = args.momentum
    bs = args.batch_size
    divide = args.divide
    divide_value = args.divide_value
    num_workers = args.num_workers
    model_type = args.model_type
    pixel_mean = [0.5]*3
    pixel_std = [0.5]*3
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    num_classes = args.num_classes
    """train_paths"""
    train_basename = os.path.basename(train_img_path)
    _, ext = os.path.splitext(train_basename)
    if ext == "":
        regex = re.compile(".*\.(jpe?g|png|gif|tif|bmp)$", re.IGNORECASE)
        train_img_paths = [file for file in glob.glob(os.path.join(train_img_path, "*.*")) if regex.match(file)]
        print("train with {} imgs".format(len(train_img_paths)))
        train_mask_paths = [os.path.join(train_mask_path, os.path.basename(file)) for file in train_img_paths]
    else:
        bs = 1
        train_img_paths = [train_img_path]
        train_mask_paths = [train_mask_path]
    """test_paths"""
    test_basename = os.path.basename(test_img_path)
    _, ext = os.path.splitext(test_basename)
    if ext == "":
        regex = re.compile(".*\.(jpe?g|png|gif|tif|bmp)$", re.IGNORECASE)
        test_img_paths = [file for file in glob.glob(os.path.join(test_img_path, "*.*")) if regex.match(file)]
        print("test with {} imgs".format(len(test_img_paths)))
        test_mask_paths = [os.path.join(test_mask_path, os.path.basename(file)) for file in test_img_paths]
    else:
        bs = 1
        test_img_paths = [test_img_path]
        test_mask_paths = [test_mask_path]
    img_size = 1024
    train_dataset = SegDataset(train_img_paths, mask_paths=train_mask_paths, mask_divide=divide,
                               divide_value=divide_value,
                               pixel_mean=pixel_mean, pixel_std=pixel_std, img_size=img_size)
    train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers)
    test_dataset = SegDataset(test_img_paths, mask_paths=test_mask_paths, mask_divide=divide, divide_value=divide_value,
                              pixel_mean=pixel_mean, pixel_std=pixel_std, img_size=img_size)
    test_dataloader = DataLoader(test_dataset, batch_size=bs, shuffle=True, num_workers=num_workers)

    if model_type == "sam":
        model = PromptSAM(model_name, checkpoint=checkpoint, num_classes=num_classes, reduction=4, upsample_times=2, groups=4)
    if args.restore_model:
        model.load_state_dict(torch.load(args.restore_model))
        print("load restore_model from{}".format(args.restore_model))


    scaler = torch.cuda.amp.grad_scaler.GradScaler()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device_type)
    if optimizer == "adamw":
        optim = opt.AdamW([{"params":model.parameters(), "initia_lr": lr}], lr=lr, weight_decay=weight_decay)
    elif optimizer == "sgd":
        optim = opt.SGD([{"params":model.parameters(), "initia_lr": lr}], lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=True)
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    scheduler = PolyLRScheduler(optim, num_images=len(train_img_paths), batch_size=bs, epochs=epochs)
    metric = Metric(num_classes=num_classes)


    '''training!!!'''
    best_iou = 0.
    train_iou_values=[]
    test_iou_values=[]
    train_losses=[]
    test_losses=[]
    #recall_values=[]
    #accuracy_values=[]
    lrs=[]
    for epoch in range(epochs):
        pred_data=[]
        target_data=[]
        losses=0
        i=0
        print("start epoch:{}".format(epoch+1))
        for i, (x, target) in enumerate(train_dataloader):
            lrs.append(optim.param_groups[0]['lr'])
            x = x.to(device)
            target = target.to(device, dtype=torch.long)
            target_2=target.squeeze(0)# 计算值为1的元素数量
            target_data.append(target_2*255)
            optim.zero_grad()
            if device_type == "cuda" and args.mix_precision:
                x = x.to(dtype=torch.float16)
                with torch.autocast(device_type=device_type, dtype=torch.float16):
                    pred = model(x)
                    #loss = loss_func(pred, target)
                    loss = seg_loss(pred, target) + ce_loss(pred, target.float())
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                x = x.to(dtype=torch.float32)
                pred = model(x)
                pred_2=pred.squeeze(0)
                mask=(pred_2[1]>pred_2[0])*255
                pred_data.append(mask)
                loss = seg_loss(pred[:,1,:,:], target) + ce_loss(pred[:,1,:,:], target.float())
                loss.backward()
                optim.step()
            losses+=loss.item()
            metric.update(torch.softmax(pred, dim=1), target)

        #每个epoch结束，更新lr
        scheduler.step()
        
        """记录指标"""
        losses/=(i+1)
        iou = np.nanmean(metric.evaluate()["iou"][1:].numpy())
        train_iou_values.append(iou)
        train_losses.append(losses)
        print("epoch-{}: average_training_iou:{},average_training_loss:{}".format(epoch+1, iou.item(),losses))
        '''evaling!!!'''
        print("evaling!")
        test_loss,test_iou=eval(model,test_dataloader)  #返回的是item!
        test_losses.append(test_loss)
        test_iou_values.append(test_iou)

        #recall_values.append(calculate_recall(pred_data,target_data))
        #accuracy_values.append(calculate_accuracy(pred_data, target_data))
        print("average_testing_iou:{},average_testing_loss:{}".format(test_iou,test_loss))
        if iou > best_iou:
            best_iou = iou
            torch.save(
                model.state_dict(), os.path.join(save_path, "{}_{}_prompt.pth".format(model_type, model_name))
            )
    #print("recall",recall_values)
    #print("accuracy",accuracy_values)
    # 将列表转换为DataFrame对象
    #df = pd.DataFrame(train_iou_values, columns=['IOU'])
     # 将DataFrame保存为CSV文件
    #df.to_csv('IOU_epoches.csv', index=False)
    """上述代码输出了指标如下：
    train/loss: iou loss
    lr
    
    """
    """绘图！"""
    plt.subplot(3, 1, 1)
    plt.plot([i + 1 for i in range(len(lrs))], lrs, marker='o', linestyle='-', label='lr')
    plt.title('lr Per times_culculate')
    plt.xlabel('time')
    plt.ylabel('learning_rate')
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot([i+1 for i in range(epochs)], train_losses, marker='o', linestyle='-', label='train_loss')
    plt.plot([i+1 for i in range(epochs)], test_losses, marker='o', linestyle='-', label='test_loss')
    plt.title('loss Per epoch')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot([i + 1 for i in range(epochs)], train_iou_values, marker='o', linestyle='-', label='train_iou')
    plt.plot([i + 1 for i in range(epochs)], test_iou_values, marker='o', linestyle='-', label='test_iou')
    plt.title('iou Per epoch')
    plt.xlabel('epoch')
    plt.ylabel('iou')
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path+"/result.png")


def eval(model,dataloader):
    device='cuda'
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    metric = Metric(num_classes=2)
    losses=0
    with torch.no_grad():
        pred_data=[]
        target_data=[]
        i=0
        for i, (x, target) in enumerate(dataloader):
            target = target.to(device, dtype=torch.long)
            target_2=target.squeeze(0)# 计算值为1的元素数量
            target_data.append(target_2*255)
            x=x.cuda()
            target.to(device)
            if device == "cuda" and args.mix_precision:
                x = x.to(dtype=torch.float16)
                with torch.autocast(device_type=device, dtype=torch.float16):
                    pred = model(x)
                    loss = seg_loss(pred, target) + ce_loss(pred, target.float())
            else:
                x = x.to(dtype=torch.float32)
                pred = model(x)
                pred_2=pred.squeeze(0)
                mask=(pred_2[1]>pred_2[0])*255
                pred_data.append(mask)
                loss = seg_loss(pred[:,1,:,:], target) + ce_loss(pred[:,1,:,:], target.float())
            losses+=loss.item()
            metric.update(torch.softmax(pred, dim=1), target)

        losses/=(i+1)
        iou = np.nanmean(metric.evaluate()["iou"][1:].numpy())
    return losses,iou.item()


if __name__ == "__main__":
    
    main(args)