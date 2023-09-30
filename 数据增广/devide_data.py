import os
import shutil

# 源文件夹
source_folder = 'CHASEDB1'

# 目标文件夹
target_folder = 'data_split'

# 创建目标文件夹及其子文件夹
os.makedirs(os.path.join(target_folder, 'img'), exist_ok=True)
os.makedirs(os.path.join(target_folder, 'mask'), exist_ok=True)
os.makedirs(os.path.join(target_folder, 'valid_mask'), exist_ok=True)

# 遍历源文件夹中的图像文件
for filename in os.listdir(source_folder):
    if filename.endswith('.jpg'):
        # 图像文件
        img_path = os.path.join(source_folder, filename)
        shutil.copy(img_path, os.path.join(target_folder, 'img', filename))
    elif filename.endswith('1stHO.png'):
        # 第一个标记的血管分割掩码
        mask_path = os.path.join(source_folder, filename)
        shutil.copy(mask_path, os.path.join(target_folder, 'mask', filename))
    elif filename.endswith('2ndHO.png'):
        # 第二个标记的血管分割掩码（用于验证）
        valid_mask_path = os.path.join(source_folder, filename)
        shutil.copy(valid_mask_path, os.path.join(target_folder, 'valid_mask', filename))

print("划分完成。")
