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
predictions_array = np.load('predictions.npy')
# 大小为:(number,2,1024,1024)
binary_images = []
for i in range(0,predictions_array.shape[0]):
    binary_image = np.where(predictions_array[i, 1] > predictions_array[i, 0], 1, 0)
    binary_images.append(binary_image)
mask = np.load("mask.npy")
y_true_data = []
for i in range(0,mask.shape[0]):
    y_true_data.append(mask[i].astype(int))
dic = 0
f1 = 0
iou = 0
BM= 0
Precision = 0
Recall = 0
print(len(binary_images))
for i in range(0,len(binary_images)):
    y_true = y_true_data[i]
    y_pred = binary_images[i]
    TN = np.sum((y_true == 0) & (y_pred == 0))
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    dic =2 * TP / (2 * TP + FN + FP) +dic
    f1  = f1_score(y_true.flatten(), y_pred.flatten())+f1
    iou = TP / float(TP + FP + FN)+iou
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
