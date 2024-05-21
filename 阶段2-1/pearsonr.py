import numpy as np
import csv

def calR(file_path):
    with open(file_path, 'r', encoding="utf-8-sig") as file:
        reader = csv.reader(file)
        rows = list(reader)
        x = list(map(float, rows[0]))
        y = list(map(float, rows[1]))
    corr = np.corrcoef(x, y)[0,1]
    print("Pearson r =", corr)


file1 = "./win_mean.csv"
file2 = "./win_cv.csv"

print("胜率和平均得分:", end="")
calR(file1)
print("胜率和变异系数:", end="")
calR(file2)