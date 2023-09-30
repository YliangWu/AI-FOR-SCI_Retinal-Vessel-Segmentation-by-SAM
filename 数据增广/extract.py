import os
import random
import shutil

def extract_images(source_dir1, target_dir1, source_dir2, target_dir2, num_images):
    # 获取源文件夹中的所有图片文件
    image_files1 = [f for f in os.listdir(source_dir1) if f.endswith('.jpg') or f.endswith('.png')]
    image_files2 = [f for f in os.listdir(source_dir2) if f.endswith('.jpg') or f.endswith('.png')]

    # 确保要抽取的数量不超过图片总数
    num_images = min(num_images, len(image_files1))

    # 随机选择指定数量的图片
    seed = random.randint(1, 50)
    random.seed(seed)
    selected_images1 = random.sample(image_files1, num_images)
    random.seed(seed)
    selected_images2 = random.sample(image_files2, num_images)

    # 将选中的图片复制到目标文件夹
    for image in selected_images1:
        source_path1 = os.path.join(source_dir1, image)
        target_path1 = os.path.join(target_dir1, image)
        source_path2 = os.path.join(source_dir2, image)
        target_path2 = os.path.join(target_dir2, image)
        shutil.copy(source_path1, target_path1)
        shutil.copy(source_path2, target_path2)

    print(f"抽取图片 {num_images} 张完成")

dataset_dir = './datasets/DRIVE/training'

transformed_training = f'{os.path.dirname(dataset_dir)}/transformed_training'
output_path = f"{transformed_training}/img"
out_mask = f"{transformed_training}/mask"

num_images_to_extract = 1

source_img_directory = f'{output_path}'
target_img_directory = f'{os.path.dirname(dataset_dir)}/extract_{num_images_to_extract}/img'
source_mask_directory = f'{out_mask}'
target_mask_directory = f'{os.path.dirname(dataset_dir)}/extract_{num_images_to_extract}/mask'
if not os.path.exists(target_img_directory): os.makedirs(target_img_directory)
if not os.path.exists(target_mask_directory): os.makedirs(target_mask_directory)
extract_images(source_img_directory, target_img_directory, source_mask_directory, target_mask_directory, num_images_to_extract)
