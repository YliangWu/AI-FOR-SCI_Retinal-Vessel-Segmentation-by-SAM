# coding:utf-8
import torch
from PIL import Image
import os
from albumentations import Compose, Resize, Normalize, ColorJitter, HorizontalFlip, VerticalFlip
from learnerable_seg import PromptSAM, PromptDiNo
import os
import glob
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import seaborn as sns
from sklearn.metrics import confusion_matrix
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
def dice_coef(y_true, y_pred,threshold):
    y_pred = (y_pred > threshold).astype(int)
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return 2*TP/(2*TP+FN+FP)

def f1(y_true, y_pred,threshold):
    y_pred = (y_pred > threshold).astype(int)
    return f1_score(y_true, y_pred)

def objective(threshold, y_true, y_pred_proba):
    y_pred = (y_pred_proba > threshold).astype(int)
    F1=f1(y_true.flatten(), y_pred.flatten(),threshold)
    Dic=dice_coef(y_true.flatten(), y_pred.flatten(),threshold)
    if F1>0.4 and Dic >0.4:
        return 0.4*F1+0.6*Dic
    else:
        return 0

def calculate_iou(y_true, y_pred_proba ,threshold):
    y_pred = (y_pred_proba > threshold).astype(int)
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    iou = TP / float(TP + FP + FN)
    return iou

def calculate_BM(y_true,y_pred_proba,threshold):
    y_pred = (y_pred_proba > threshold).astype(int)
    TN = np.sum((y_true == 0) & (y_pred == 0))
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    BM = TP/(TP+FN)+TN/(TN+FP)-1
    return BM

def calculate_Recall(y_true,y_pred_proba,threshold):
    y_pred = (y_pred_proba > threshold).astype(int)
    TN = np.sum((y_true == 0) & (y_pred == 0))
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return TP/(TP+FN)

def calculate_Precision(y_true,y_pred_proba,threshold):
    y_pred = (y_pred_proba > threshold).astype(int)
    TN = np.sum((y_true == 0) & (y_pred == 0))
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return TP/(TP+FP)


def main():
    pixel_mean = [0.5] * 3
    pixel_std = [0.5] * 3
    predictions = []
    y_true_data = []
    model = PromptSAM('vit_b', checkpoint='./weights/sam_vit_b_01ec64.pth', num_classes=2, reduction=4, upsample_times=2,
                          groups=4)
    #加载模型
    checkpoint = torch.load('./liao/sam_vit_b_prompt_20230924-2238.pth')
    model.load_state_dict(checkpoint)

    device = torch.device('cpu')
    model.to(device)
    # 遍历test文件夹中的所有.jpg文件
    for img_path in glob.glob("./valid/*.jpg"):
        img = Image.open(img_path).convert("RGB")  # 使用RGB图
        img = np.asarray(img)

        # 数据处理
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

        x = torch.from_numpy(x)
        x = x.unsqueeze(0)
        x = x.to(device)

        with torch.no_grad():
            pred = model(x)
        mask = pred.squeeze(0)
        y_true_data.append(x.numpy())
        # 将预测结果添加到列表中
        predictions.append(mask.numpy())
    # 参数初始化
    num_population = 20
    num_generations = 20
    bits_per_variable = 10
    elite_size = 5
    variables_range = np.array([[0.5, 0.8]])
    # 定义变量个数
    num_variables = 1
    # 初始化种群
    population = np.random.randint(0, 2, (num_population, num_variables * bits_per_variable))
    # 在进化开始前定义初始的交叉率和变异率
    initial_crossover_rate = 0.8
    initial_mutation_rate = 0.2

    def solve(X):
        threshold = X
        w = 0
        for i in range(len(predictions)):
            w = objective(threshold, y_true_data[i], predictions[i])+w
        return w/(len(predictions))

    # 定义适应度函数
    def fitness(individual):
        # 解码
        threshold = decode(individual)
        w = 0
        for i in range(len(predictions)):
            w = objective(threshold, y_true_data[i], predictions[i]) + w
        return w / (len(predictions))

    # 定义解码函数
    def decode(individual):
        # 将二进制编码的个体转换为实数
        decoded_individual = []
        for i in range(num_variables):
            binary = individual[i * bits_per_variable:(i + 1) * bits_per_variable]
            value = binary.dot(2 ** np.arange(binary.size)[::-1])
            value = value / (2 ** bits_per_variable - 1) * (variables_range[i][1] - variables_range[i][0]) + \
                    variables_range[i][0]
            decoded_individual.append(value)
        return np.array(decoded_individual)

    # 进化开始
    # 在每一代中更新交叉率和变异率

    # 在每一代中保存最优适应度和平均适应度
    best_fitnesses = []
    avg_fitnesses = []
    for generation in range(num_generations):
        print(generation)
        crossover_rate = initial_crossover_rate * (1 - generation / num_generations)
        mutation_rate = initial_mutation_rate * (1 - generation / num_generations)
        # 评估当前种群
        fitness_values = np.array([fitness(ind) for ind in population])
        best_fitness = np.max(fitness_values)
        avg_fitness = np.mean(fitness_values)
        best_fitnesses.append(best_fitness)
        avg_fitnesses.append(avg_fitness)
        # 精英保留
        elites_indices = fitness_values.argsort()[-elite_size:]
        elites = population[elites_indices]

        # 选择
        selected_indices = np.random.choice(np.arange(num_population), size=num_population - elite_size, replace=True,
                                            p=fitness_values / np.sum(fitness_values))
        population = population[selected_indices]

        # 交叉
        for i in range(0, num_population - elite_size - 1, 2):  # -1 is added here to avoid index out of bound error
            if np.random.rand() < crossover_rate:
                crossover_point = np.random.randint(num_variables * bits_per_variable)
                population[i, crossover_point:], population[i + 1, crossover_point:] = population[i + 1,
                                                                                       crossover_point:].copy(), population[
                                                                                                                 i,
                                                                                                                 crossover_point:].copy()

        # 变异
        for i in range(num_population - elite_size):
            if np.random.rand() < mutation_rate:
                mutation_point = np.random.randint(num_variables * bits_per_variable)
                population[i, mutation_point] = 1 if population[i, mutation_point] == 0 else 0

        # 将精英个体加回种群
        population = np.vstack((population, elites))

    # 找出最优个体
    fitness_values = np.array([fitness(ind) for ind in population])
    best_individual = decode(population[np.argmax(fitness_values)])
    solve(best_individual)
    print('Best Individual: ', best_individual)
    # 绘制每一代的最优适应度和平均适应度
    plt.figure(figsize=(10, 6))
    plt.plot(best_fitnesses, label='Best Fitness')
    plt.plot(avg_fitnesses, label='Average Fitness')
    plt.xlabel('Generation', fontsize=14)
    plt.ylabel('Fitness', fontsize=14)
    plt.title('Best and Average Fitness over Generations', fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.show()
    # 在此最优阈值下，计算模型在该验证集上的各指标和ROC曲线以及PR曲线
    F1 = 0
    Iou = 0
    dice = 0
    BM = 0
    Recall = 0
    Precision = 0

    for i in range(len(predictions)):
        F1 = f1(y_true_data[i],predictions[i],best_individual) + F1
        Iou = calculate_iou((y_true_data[i],predictions[i],best_individual)) + Iou
        dice = dice_coef(y_true_data[i],predictions[i],best_individual) + dice
        BM = calculate_BM(y_true_data[i],predictions[i],best_individual) + BM
        Recall = calculate_Recall(y_true_data[i],predictions[i],best_individual) + Recall
        Precision = calculate_Precision(y_true_data[i],predictions[i],best_individual) + Precision

    F1=F1/len(predictions)
    Iou=Iou/len(predictions)
    dice=dice/len(predictions)
    BM=BM/len(predictions)
    Recall=Recall/len(predictions)
    Precision=Precision/len(Precision)
    print(F1,Iou,dice,BM,Recall,Precision)
    # 在此验证集下绘制ROC曲线和PR曲线
    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(y_true, pred_blood)
    roc_auc = auc(fpr, tpr)

    # 计算PR曲线
    precision, recall, _ = precision_recall_curve(y_true, pred_blood)
    pr_auc = auc(recall, precision)

    # 设置Seaborn图形风格
    sns.set(style="whitegrid")

    plt.figure(figsize=(10, 5))

    # 绘制ROC曲线
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # 绘制PR曲线
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='deepskyblue', lw=2, label='PR curve (area = %0.2f)' % pr_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()