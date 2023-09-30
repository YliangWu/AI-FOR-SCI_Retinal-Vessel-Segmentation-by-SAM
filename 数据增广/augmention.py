import albumentations as A
import torch
import math
import random
import os
import cv2
import shutil
import numpy as np
import argparse
from torchvision import transforms
from PIL import Image


def make_odd(num):
    num = math.ceil(num)
    if num % 2 == 0:
        num += 1
    return num


def med_augment(data_path, name, level, number_branch, mask_i=True, shield=False):
    if mask_i:
        image_path = f"{data_path}{name}"
        mask_path = f"{data_path}valid_mask"
        output_path = f"{os.path.dirname(os.path.dirname(data_path))}/medaugment/{name}/"
        out_mask = f"{os.path.dirname(os.path.dirname(data_path))}/medaugment/valid_mask/"
    else:
        image_path = data_path + name
        output_path = f"{os.path.dirname(os.path.dirname(os.path.dirname(data_path)))}/medaugment/training/{name}/"

    transform = A.Compose([
        A.ColorJitter(brightness=0.04 * level, contrast=0, saturation=0, hue=0, p=0.2 * level),
        A.ColorJitter(brightness=0, contrast=0.04 * level, saturation=0, hue=0, p=0.2 * level),
        A.Posterize(num_bits=math.floor(8 - 0.8 * level), p=0.2 * level),
        A.Sharpen(alpha=(0.04 * level, 0.1 * level), lightness=(1, 1), p=0.2 * level),
        A.GaussianBlur(blur_limit=(3, make_odd(3 + 0.8 * level)), p=0.2 * level),
        A.GaussNoise(var_limit=(2 * level, 10 * level), mean=0, per_channel=True, p=0.2 * level),
        A.Rotate(limit=4 * level, interpolation=1, border_mode=0, value=0, mask_value=None, rotate_method='largest_box',
                 crop_border=False, p=0.2 * level),
        A.HorizontalFlip(p=0.2 * level),
        A.VerticalFlip(p=0.2 * level),
        A.Affine(scale=(1 - 0.04 * level, 1 + 0.04 * level), translate_percent=None, translate_px=None, rotate=None,
                 shear=None, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False,
                 keep_ratio=True, p=0.2 * level),
        A.Affine(scale=None, translate_percent=None, translate_px=None, rotate=None,
                 shear={'x': (0, 2 * level), 'y': (0, 0)}
                 , interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False,
                 keep_ratio=True, p=0.2 * level),  # x
        A.Affine(scale=None, translate_percent=None, translate_px=None, rotate=None,
                 shear={'x': (0, 0), 'y': (0, 2 * level)}
                 , interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False,
                 keep_ratio=True, p=0.2 * level),
        A.Affine(scale=None, translate_percent={'x': (0, 0.02 * level), 'y': (0, 0)}, translate_px=None, rotate=None,
                 shear=None, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False,
                 keep_ratio=True, p=0.2 * level),
        A.Affine(scale=None, translate_percent={'x': (0, 0), 'y': (0, 0.02 * level)}, translate_px=None, rotate=None,
                 shear=None, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False,
                 keep_ratio=True, p=0.2 * level)
    ])

    for j, file_name in enumerate(os.listdir(image_path)):
        if file_name.endswith(".png") or file_name.endswith(".jpg"):
            file_path = os.path.join(image_path, file_name)
            file_n, file_s = file_name.split(".")[0], file_name.split(".")[1]
            image = cv2.imread(file_path)
            if mask_i:
                mask = cv2.imread(f"{mask_path}/{file_n}_2ndHO.png")
                # print(f"{mask_path}/{file_n}_2ndHO.png")

            strategy = [(1, 2), (0, 3), (0, 2), (1, 1)]
            for i in range(number_branch):
                if number_branch != 4:
                    employ = random.choice(strategy)
                else:
                    index = random.randrange(len(strategy))
                    employ = strategy.pop(index)
                level, shape = random.sample(transform[:6], employ[0]), random.sample(transform[6:], employ[1])
                img_transform = A.Compose([*level, *shape])
                random.shuffle(img_transform.transforms)
                if not os.path.exists(output_path): os.makedirs(output_path)
                if not os.path.exists(out_mask): os.makedirs(out_mask)
                if mask_i:
                    transformed = img_transform(image=image, mask=mask)
                    transformed_image, transformed_mask = transformed['image'], transformed['mask']
                    # print(image.shape, '\n', mask.shape, '\n', transformed_mask.shape)
                    cv2.imwrite(f"{output_path}{file_n}_{i + 1}.{file_s}", transformed_image)
                    cv2.imwrite(f"{out_mask}{file_n}_{i + 1}.{file_s}", transformed_mask)
                else:
                    transformed = img_transform(image=image)
                    transformed_image = transformed['image']
                    cv2.imwrite(f"{output_path}/{file_n}_{i + 1}.{file_s}", transformed_image)
                if not shield:
                    cv2.imwrite(f"{output_path}/{file_n}_{number_branch + 1}.{file_s}", image)
                    if mask_i: cv2.imwrite(f"{out_mask}/{file_n}_{number_branch + 1}.{file_s}", mask)

def generate_datasets(dataset, seed, level, number_branch):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    print('Executing data augmentation for image classification...')
    data_path = f"./datasets/{dataset}/training/"
    folder_path = f"./datasets/{dataset}/"

    name = 'img'
    med_augment(data_path, name, level, number_branch)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_argument_group()
    group.add_argument('--dataset', default='CHASEDB1')
    group.add_argument('--level', help='Augmentation level', default=5, type=int, metavar='INT')
    group.add_argument('--number_branch', help='Number of branch', default=99, type=int, metavar='INT')
    group.add_argument('--seed', help='Seed', default=8, type=int, metavar='INT')
    args = parser.parse_args()
    generate_datasets(**vars(args))


if __name__ == '__main__':
    main()