import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

# 构建特征矩阵
data = np.array([
    [29, 4, 5, 0, 0],
    [25, 1, 9, 3, 0],
    [10, 10, 0, 0, 0],
    [9, 11, 6, 0, 2],
    [13, 2, 4, 0, 0],
    [12, 2, 2, 0, 0],
    [10, 7, 2, 0, 0],
    [13, 4, 0, 0, 0],
    [23, 16, 6, 0, 5],
    [18, 3, 3, 0, 0],
    [16, 2, 6, 0, 0],
    [30, 3, 1, 0, 0],
    [24, 17, 4, 0, 4],
    [12, 10, 4, 0, 3],
    [18, 6, 0, 0, 0],
    [13, 2, 0, 0, 0],
    [33, 4, 5, 0, 0],
    [15, 12, 4, 0, 2],
    [16, 4, 4, 1, 0],
    [15, 2, 2, 0, 0]
])

# 定义权重
weights = np.array([0.4, 0.2, 0.2, 0.1, 0.1])

# 加权处理
weighted_data = data * weights

# 标准化加权后的数据
weighted_data_std = (weighted_data - np.mean(weighted_data, axis=0)) / np.std(weighted_data, axis=0)

# 创建PCA对象，指定要保留的主成分个数为2
pca = PCA(n_components=2)

# 进行PCA降维
pca_data = pca.fit_transform(weighted_data_std)

# 计算每个球员的降维后特征向量与坐标原点之间的欧氏距离
distances = np.linalg.norm(pca_data, axis=1)

# 选取距离最近的前5个球员作为核心球员
core_players_indices = np.argsort(distances)[:5]

# 输出核心球员的索引和距离
for i in core_players_indices:
    print(f"Player {i+1}: Distance = {distances[i]}")

# 绘制降维后的数据散点图
plt.scatter(pca_data[:, 0], pca_data[:, 1])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Visualization')
plt.show()