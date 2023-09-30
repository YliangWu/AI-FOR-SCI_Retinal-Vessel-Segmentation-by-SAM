# coding:utf-8
import torch
from PIL import Image
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from albumentations import Compose, Resize, Normalize, ColorJitter, HorizontalFlip, VerticalFlip
from learnerable_seg import PromptSAM, PromptDiNo

"""read_me"""
"目前很呆瓜的单张图像输入，输出"
"修改img_path"
#
weight_path='sam_vit_b_prompt.pth'

def main():
    pixel_mean = [0.5] * 3
    pixel_std = [0.5] * 3
    img_path="Image_01L.jpg"
    model = PromptSAM('vit_b', checkpoint='./weights/sam_vit_b_01ec64.pth', num_classes=2, reduction=4, upsample_times=2,
                          groups=4)
    #加载模型
    checkpoint = torch.load(weight_path)
    model.load_state_dict(checkpoint)

    device = torch.device('cpu')
    model.to(device)
    img = Image.open(img_path).convert("RGB")  # 使用RGB图
    img = np.asarray(img)
    #数据处理
    transform = Compose(
        [
            ColorJitter(),
            VerticalFlip(),
            HorizontalFlip(),
            Resize(1024, 1024),
            Normalize(mean=pixel_mean, std=pixel_std)
        ]
    )
    aug_data = transform(image=img)
    x = aug_data["image"]
    if img.ndim == 3:
        x = np.transpose(x, axes=[2, 0, 1])
    elif img.ndim == 2:
        x = np.expand_dims(x, axis=0)
    x=torch.from_numpy(x)
    x=x.unsqueeze(0)
    x = x.to(device)
    with torch.no_grad():
        pred = model(x)
    pred=pred.squeeze(0) # 【1,102】
    #pred 前向传播的输出，  【0】表示属于背景的概率 【1】表示属于血管的概率、
    mask=(pred[1]>pred[0])*255 # 提取概率更大的那一个， 转换成 0-255 的二值图像
    output_image = Image.fromarray(np.uint8(mask))
    output_image.save('output_image.jpg')
    print("image out to"+"output_image.jpg")


if __name__ == "__main__":
    main()