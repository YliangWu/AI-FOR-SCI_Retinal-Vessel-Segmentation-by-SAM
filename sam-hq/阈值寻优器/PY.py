import matplotlib.pyplot as plt
import seaborn as sns

# 模型和指标数据
models = ['One Shoot', 'Two Shoot', 'Five Shoot', 'Many Shoot']
f1_scores = [0.69, 0.711, 0.687, 0.76]
iou_scores = [0.52, 0.553, 0.527, 0.61]
dic_scores = [0.685, 0.711, 0.688, 0.76]
bm_scores = [0.6, 0.65, 0.602, 0.67]
recall_scores = [0.614, 0.667, 0.618, 0.68]
precision_scores = [0.787, 0.772, 0.789, 0.85]
AUROC = [0.765,0.796,0.826,0.864]
# 设置图形风格
sns.set(style='whitegrid')
plt.figure(figsize=(8, 6))

# 定义颜色映射
colors = ['lightblue', 'lightgreen', 'lightpink', 'lightyellow']

# 绘制条形图
x = range(len(models))
bar_width = 0.12
opacity = 0.8

plt.bar(x, f1_scores, bar_width, alpha=opacity, color=colors[0], label='F1')
plt.bar([i + bar_width for i in x], iou_scores, bar_width, alpha=opacity, color=colors[1], label='IOU')
plt.bar([i + 2 * bar_width for i in x], dic_scores, bar_width, alpha=opacity, color=colors[2], label='DIC')
plt.bar([i + 3 * bar_width for i in x], bm_scores, bar_width, alpha=opacity, color=colors[3], label='BM')
plt.bar([i + 4 * bar_width for i in x], recall_scores, bar_width, alpha=opacity, color=colors[0], label='Recall')
plt.bar([i + 5 * bar_width for i in x], precision_scores, bar_width, alpha=opacity, color=colors[1], label='Precision')
# 绘制AUROC条形图
plt.bar([i + 6 * bar_width for i in x], AUROC, bar_width, alpha=opacity, color=colors[2], label='AUROC')

# 设置坐标轴标签和标题
plt.xlabel('Models')
plt.ylabel('Scores')
plt.title('Comparison of Model Performance on Different Metrics')
plt.xticks([i + 2.5 * bar_width for i in x], models)
plt.legend(loc='lower right')

# 添加数值标签
for i, score in enumerate(f1_scores + iou_scores + dic_scores + bm_scores + recall_scores + precision_scores + AUROC):
    plt.text(i % len(models) + (i // len(models)) * bar_width - 0.05, score + 0.01, str(score), fontsize=8)


# 显示图形
plt.show()