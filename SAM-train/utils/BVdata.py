from PIL import Image
import torch
import numpy as np
from albumentations import Compose, Resize, Normalize, ColorJitter, HorizontalFlip, VerticalFlip

class SegDataset:
    def __init__(self, img_paths, mask_paths,
                 mask_divide=False, divide_value=255,
                 pixel_mean=[0.5] * 3, pixel_std=[0.5] * 3,
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
        img = Image.open(img_path).convert("RGB")  # 使用RGB图
        img = np.asarray(img)
        mask = Image.open(mask_path).convert("L")  # 使用灰度图
        mask = np.asarray(mask)
        if self.mask_divide:
            mask = mask // self.divide_value  # 0-1化
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
        target = np.expand_dims(target, axis=0)
        return torch.from_numpy(x), torch.from_numpy(target)
