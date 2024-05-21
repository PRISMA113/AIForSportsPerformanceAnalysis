from scipy.stats import chi2_contingency
import csv
import pandas as pd
import numpy as np

def normalize(x):
    x_normalized = [(val - min(x)) / (max(x) - min(x)) + 0.01 for val in x]
#    print(x_normalized)
    return x_normalized


def calChi2(file_path, alpha):
    with open(file_path, 'r', encoding="utf-8-sig") as file:
        reader = csv.reader(file)
        rows = list(reader)
        x = list(map(float, rows[0]))
        y = list(map(float, rows[1]))

    x = normalize(x)
    y = normalize(y)
    observed = np.array([x, y])

    # 执行卡方独立性检验
    chi2, p, dof, expected = chi2_contingency(observed)

    # 输出结果
    print("卡方统计量:", chi2)
    print("P值:", p)
    # print("自由度:", dof)
    # print("期望频数:", expected)

    # 根据P值判断独立性
    if p < alpha:
        print("数据不独立")
    else:
        print("数据独立")


file1 = "./win_mean.csv"
file2 = "./win_cv.csv"

print("胜率和平均得分:", end="")
calChi2(file1, 0.05)
print("胜率和变异系数:", end="")
calChi2(file2, 0.05)
