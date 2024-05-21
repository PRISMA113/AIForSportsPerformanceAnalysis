import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# 创建数据框
data = pd.DataFrame({
    '总出手次数': [37, 24, 17, 13, 11, 108, 120, 109, 125, 115, 88, 86, 84],
    '失误次数': [14, 10, 9, 8, 7, 15, 17, 18, 25, 30, 13, 11, 14],
    '罚球次数': [8, 10, 7, 9, 9, 19, 19, 15, 14, 11, 18, 18, 22],
    '总进攻篮板数': [48, 52, 46, 42, 37, 46, 44, 42, 39, 37, 41, 51, 37],
    '总得分': [115, 121, 113, 111, 108, 119, 108, 85, 101, 102, 106, 103, 102],
    '胜负': [0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0]
})

# 添加截距项
data = sm.add_constant(data)

# 拟合多元线性回归模型
model = sm.OLS(data['胜负'], data[['const', '总出手次数', '失误次数', '罚球次数', '总进攻篮板数', '总得分']])
results = model.fit()

# 打印回归结果
print(results.summary())

# 绘制每个变量的回归曲线
sns.pairplot(data, x_vars=['总出手次数', '失误次数', '罚球次数', '总进攻篮板数', '总得分'], y_vars='胜负', kind='reg')
plt.show()

