"""
代码说明：
该代码使用re库，正则化匹配方法，提取nohup.out输出日志的有用信息，进行绘图
使用仅需要修改patern就行，可以用gpt帮忙


"""


import re
import matplotlib.pyplot as plt

def plot_log(path):
   # 输入日志文件路径
   log_file = "./nohup.out"
   # 定义正则表达式，用于匹配 loss 值
   # 打开日志文件并读取内容
   with open(log_file, 'r') as f:
      log_text = f.read()
   # 使用正则表达式查找所有匹配的 loss 值
   # 使用正则表达式提取loss信息
   loss_pattern = r'train_Averaged stats: training_loss: ([0-9.]+)'
   loss_values = re.findall(loss_pattern, log_text)

   lr_pattern = r'learning rate:  \s+([0-9.]+)'
   lr_values = re.findall(lr_pattern, log_text)

   iou_pattern = r'Averaged stats: val_iou_0:\s+([0-9.]+)'
   iou_values = re.findall(iou_pattern, log_text)


   iou_values = [float(value) for value in iou_values]


   # 转换为浮点数
   loss_values = [float(value) for value in loss_values]
   lr_values = [float(value) for value in lr_values]

   # 计算每个epoch的平均loss
   num_epochs = 30  # 你的示例中有4个epoch


   # 创建epoch列表，用于绘图
   epochs = [f'Epoch {i+1}' for i in range(num_epochs)]

   # 绘制loss图表
   plt.figure(figsize=(12, 6))
   plt.subplot(3, 1, 1)
   plt.plot([i+1 for i in range(len(loss_values))], loss_values, marker='o', linestyle='-')
   plt.title('Average Loss Per Epoch')
   plt.xlabel('Epoch')
   plt.ylabel('Average Loss')
   plt.grid(True)

   # 绘制learning rate图表
   plt.subplot(3, 1, 2)
   plt.plot([i+1 for i in range(len(lr_values))], lr_values, marker='o', linestyle='-')
   plt.title('Learning Rate Per Epoch')
   plt.xlabel('Epoch')
   plt.ylabel('Learning Rate')
   plt.grid(True)

   # 绘制IOU图表
   plt.subplot(3, 1, 3)
   plt.plot([i+1 for i in range(len(iou_values))], iou_values, marker='o', linestyle='-', label='val_iou_0')
   plt.title('IOU Metrics Per Epoch')
   plt.xlabel('Epoch')
   plt.ylabel('IOU')
   plt.legend()
   plt.grid(True)

   plt.tight_layout()
   plt.savefig(path+"/res.png")






