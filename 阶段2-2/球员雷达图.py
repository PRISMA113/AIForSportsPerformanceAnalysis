import matplotlib.pyplot as plt
import numpy as np

# 球员数据
players = [
    ['Klay Thompson', 29, 4, 5, 0, 0],
    ['Stephen Curry', 25, 1, 9, 3, 0],
    ['Zaza Pachulia', 10, 10, 0, 0, 0],
    ['Draymond Green', 9, 11, 6, 0, 2],
    ['Elfrid Payton', 13, 2, 4, 0, 0],
    ['Evan Fournier', 12, 2, 2, 0, 0],
    ['Nikola Vucevic', 10, 7, 2, 0, 0],
    ['Jeff Green', 13, 4, 0, 0, 0],
    ['Al Horford', 23, 16, 6, 0, 5],
    ['Dennis Schroder', 18, 3, 3, 0, 0],
    ['Jeff Teague', 16, 2, 6, 0, 0],
    ['Derrick Rose', 30, 3, 1, 0, 0],
    ['Pau Gasol', 24, 17, 4, 0, 4],
    ['Joakim Noah', 12, 10, 4, 0, 3],
    ['Taj Gibson', 18, 6, 0, 0, 0],
    ['Doug McDermott', 13, 2, 0, 0, 0],
    ['Stephen Curry', 33, 4, 5, 0, 0],
    ['Draymond Green', 15, 12, 4, 0, 2],
    ['Andre Iguodala', 16, 4, 4, 1, 0],
    ['Klay Thompson', 15, 2, 2, 0, 0]
]

# 提取数据
labels = ['得分', '篮板', '助攻', '抢断', '盖帽']
num_players = len(players)

# 创建雷达图的角度
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]

# 创建雷达图
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, polar=True)

# 绘制雷达图数据
for i, (player, *data) in enumerate(players):
    values = data + data[:1]
    ax.plot(angles, values, marker='o', label=player)

# 添加每个标签的角度位置
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)

# 设置每个标签的字体大小
ax.tick_params(axis='x', labelsize=10)

# 添加图例
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))

# 显示雷达图
plt.show()
