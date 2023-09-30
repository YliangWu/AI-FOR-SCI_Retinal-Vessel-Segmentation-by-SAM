import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 模型和指标数据
models = ['LPSAM', 'SAM-HQ']
indicators = ['F1', 'IOU', 'DIC', 'BM', 'Recall', 'Precision','AUROC']
scores = np.array([[0.228, 0.76],
                   [0.159, 0.61],
                   [0.229, 0.76],
                   [0.162, 0.67],
                   [0.209, 0.68],
                   [0.254, 0.85],
                   [0.264, 0.864]])

# 设置图形风格
sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))

# 定义颜色映射
colors = sns.color_palette("pastel")

# 绘制条形图
bar_width = 0.35
opacity = 0.8
index = np.arange(len(indicators))

for i, model in enumerate(models):
    plt.barh(index + i * bar_width, scores[:, i], bar_width, alpha=opacity, color=colors[i], label=model)

# 设置坐标轴标签和标题
plt.xlabel('Scores')
plt.ylabel('Indicators')
plt.title('Comparison of Model Performance ')
plt.yticks(index + bar_width, indicators)
plt.legend(loc='lower right')

# 添加数值标签
for i, indicator in enumerate(indicators):
    for j, model in enumerate(models):
        plt.text(scores[i, j] + 0.01, index[i] + j * bar_width, str(scores[i, j]), fontsize=8)

# 显示图形
plt.show()