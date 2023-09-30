# coding:utf-8
import torch
from PIL import Image
import os
from albumentations import Compose, Resize, Normalize, ColorJitter, HorizontalFlip, VerticalFlip
import os
import glob
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import seaborn as sns
from sklearn.metrics import confusion_matrix
# 读取.npy文件
y_true_data = np.load('ground_truth.npy')
threshold = 2
y_true_data = np.where(y_true_data > threshold, 1, 0)
binary_images = np.load("masks.npy")
dic = 0
f1 = 0
iou = 0
BM= 0
Precision = 0
Recall = 0
IOU = []
for i in range(0,y_true_data.shape[0]):
    y_true = y_true_data[i]
    y_pred = binary_images[i]
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    TN = np.sum((y_true == 0) & (y_pred == 0))
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    dic =2 * TP / (2 * TP + FN + FP) +dic
    f1  = f1_score(y_true.flatten(), y_pred.flatten())+f1
    iou = TP / float(TP + FP + FN)+iou
    IOU.append(TP / float(TP + FP + FN))
    BM = TP / (TP + FN) + TN / (TN + FP) - 1+BM
    Precision = TP / (TP + FP)+Precision
    Recall = TP / (TP + FN)+Recall
F1=f1/len(binary_images)
Iou=iou/len(binary_images)
dic=dic/len(binary_images)
BM=BM/len(binary_images)
Recall=Recall/len(binary_images)
Precision=Precision/len(binary_images)
print(F1,Iou,dic,BM,Recall,Precision)
y_true_flattened = y_true_data.flatten()
y_pred_flattened = binary_images.flatten()
# 设置图形风格
sns.set(style="white", font_scale=1.2)
# 绘制直方图
plt.figure(figsize=(8, 6))
sns.histplot(IOU, bins=10, kde=True, color='lightblue')
# 设置坐标轴标签和标题
plt.xlabel('IOU')
plt.ylabel('Frequency')
plt.title('Distribution of IOU Values')
# 添加网格线
plt.grid(True, linestyle='--', alpha=0.5)
# 显示图形
plt.show()



