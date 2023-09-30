import numpy as np
import os
import random
from shutil import copyfile
import torch

dataset_dir = './datasets/CHASEDB1/medaugment'
train_ratio = 0.7  # 训练集比例
test_ratio = 0.2   # 测试集比例
val_ratio = 0.1    # 验证集比例

train_dir = f'{os.path.dirname(dataset_dir)}/splited/train'
test_dir = f'{os.path.dirname(dataset_dir)}/splited/test'
val_dir = f'{os.path.dirname(dataset_dir)}/splited/validation'

os.makedirs(os.path.join(train_dir, 'img'), exist_ok=True)
os.makedirs(os.path.join(train_dir, 'valid_mask'), exist_ok=True)
os.makedirs(os.path.join(test_dir, 'img'), exist_ok=True)
os.makedirs(os.path.join(test_dir, 'valid_mask'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'img'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'valid_mask'), exist_ok=True)

image_files = os.listdir(os.path.join(dataset_dir, 'img'))
image_files = sorted(image_files)
mask_files = os.listdir(os.path.join(dataset_dir, 'valid_mask'))
mask_files = sorted(mask_files)
#print(image_files[70])
#print(image_files[71])
#print(image_files[69])
#print(mask_files[70])
#print(mask_files[71])
#print(mask_files[69])

# 使用相同的随机种子打乱图片和掩膜的顺序
seed = random.randint(1, 50)
random.seed(seed)
array1 = list(range(0, 280))
# print(array1)
random.shuffle(array1)

total_samples = len(image_files)
train_samples = int(total_samples * train_ratio)
test_samples = int(total_samples * test_ratio)
val_samples = total_samples - train_samples - test_samples
print(total_samples, train_samples, test_samples, val_samples)

# 划分训练集
for i in range(train_samples):
    # print(image_files[i], mask_files[i])
    src_image = os.path.join(dataset_dir, 'img', image_files[array1[i]])
    src_mask = os.path.join(dataset_dir, 'valid_mask', mask_files[array1[i]])
    dst_image = os.path.join(train_dir, 'img', image_files[array1[i]])
    dst_mask = os.path.join(train_dir, 'valid_mask', mask_files[array1[i]])
    #print(src_mask)
    #print(src_image)
    # print(array1[i])
    #print(mask_files[array1[i]])
    #print(image_files[array1[i]])
    copyfile(src_image, dst_image)
    copyfile(src_mask, dst_mask)

# 划分测试集
for i in range(train_samples, train_samples + test_samples):
    src_image = os.path.join(dataset_dir, 'img', image_files[array1[i]])
    src_mask = os.path.join(dataset_dir, 'valid_mask', mask_files[array1[i]])
    dst_image = os.path.join(test_dir, 'img', image_files[array1[i]])
    dst_mask = os.path.join(test_dir, 'valid_mask', mask_files[array1[i]])
    copyfile(src_image, dst_image)
    copyfile(src_mask, dst_mask)

# 划分验证集
for i in range(train_samples + test_samples, total_samples):
    src_image = os.path.join(dataset_dir, 'img', image_files[array1[i]])
    src_mask = os.path.join(dataset_dir, 'valid_mask', mask_files[array1[i]])
    dst_image = os.path.join(val_dir, 'img', image_files[array1[i]])
    dst_mask = os.path.join(val_dir, 'valid_mask', mask_files[array1[i]])
    copyfile(src_image, dst_image)
    copyfile(src_mask, dst_mask)


