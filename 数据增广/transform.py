import os
from PIL import Image

dataset_dir = './datasets/DRIVE/training'

transformed_training = f'{os.path.dirname(dataset_dir)}/transformed_training'
output_path = f"{transformed_training}/img"
out_mask = f"{transformed_training}/mask"
if not os.path.exists(output_path): os.makedirs(output_path)
if not os.path.exists(out_mask): os.makedirs(out_mask)

image_path = f"{dataset_dir}/images"
mask_path = f"{dataset_dir}/1st_manual"

'''
# 打开TIFF图像
tiff_image = Image.open('input.tif')

# 将TIFF转换为JPEG
tiff_image.save('output.jpg', 'JPEG')

# 打开GIF图片
gif_image = Image.open('input.gif')

# 将GIF转换为JPG
jpg_image = gif_image.convert('RGB')

# 保存为JPG格式
jpg_image.save('output.jpg')
'''
for i, file_name in enumerate(os.listdir(image_path)):
    #if file_name.endswith(".png") or file_name.endswith(".jpg"):
    file_path = os.path.join(image_path, file_name)
    file_n, file_s = file_name.split(".")[0], file_name.split(".")[1]
    file_mask_n, x = file_name.split("_")[0], file_name.split("_")[1]
        # image = cv2.imread(file_path)
    print(file_mask_n)
    mask_file_path = os.path.join(mask_path, f"{file_mask_n}_manual1.gif")
    # 打开TIFF图像
    tiff_image = Image.open(f'{file_path}')
    # 将TIFF转换为JPEG
    tiff_image.save(f"{output_path}/{file_n}.jpg", 'JPEG')
    # 打开GIF图片
    gif_image = Image.open(f'{mask_file_path}')
    # 将GIF转换为JPG
    jpg_image = gif_image.convert('RGB')
    # 保存为JPG格式
    jpg_image.save(f"{out_mask}/{file_n}.jpg", 'JPEG')