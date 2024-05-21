import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# 创建比赛数据的DataFrame
data = {
    '对手': ['Houston Rockets', 'Milwaukee Bucks', 'Orlando Magic', 'Charlotte Hornets',
            'Portland Trail Blazers', 'Brooklyn Nets', 'Portland Trail Blazers',
            'Los Angeles Lakers', 'Minnesota Timberwolves', 'Detroit Pistons',
            'Atlanta Hawks', 'Utah Jazz', 'Detroit Pistons'],
    '总出手次数': [37, 24, 17, 13, 11, 108, 120, 109, 125, 115, 88, 86, 84],
    '失误次数': [14, 10, 9, 8, 7, 15, 17, 18, 25, 30, 13, 11, 14],
    '罚球次数': [8, 10, 7, 9, 9, 19, 19, 15, 14, 11, 18, 18, 22],
    '总进攻篮板数': [48, 52, 46, 42, 37, 46, 44, 42, 39, 37, 41, 51, 37],
    '总得分': [115, 121, 113, 111, 108, 119, 108, 85, 101, 102, 106, 103, 102],
    '胜负': ['负', '负', '胜', '胜', '胜', '胜', '负', '负', '胜', '负', '胜', '负', '负']
}
df = pd.DataFrame(data)

# 计算进攻效率
df['进攻效率'] = df['总得分'] / (df['总出手次数'] - df['总进攻篮板数'] + df['失误次数'] + df['罚球次数'])

# 将胜负转换为二进制值（1代表胜利，0代表失败）
df['胜负'] = df['胜负'].apply(lambda x: 1 if x == '胜' else 0)

# 使用逻辑回归分析进攻效率对胜负的影响
X = df['进攻效率']  # 自变量：进攻效率
Y = df['胜负']     # 因变量：胜负

# 添加截距项
X = sm.add_constant(X)

# 拟合逻辑回归模型
model = sm.Logit(Y, X)
results = model.fit()

# 输出回归结果
print(results.summary())

# 绘制逻辑回归拟合曲线和散点图
plt.scatter(df['进攻效率'], df['胜负'], label='Actual Data')

# 生成逻辑回归拟合曲线的预测值
x_vals = np.linspace(df['进攻效率'].min(), df['进攻效率'].max(), 100)
x_vals = sm.add_constant(x_vals)
y_vals = results.predict(x_vals)

# 绘制逻辑回归拟合曲线
plt.plot(x_vals[:, 1], y_vals, color='red', label='Logistic Regression Fit')
plt.xlabel('进攻效率')
plt.ylabel('胜负')
plt.title('进攻效率对比赛胜负的关系')
plt.legend()
plt.show()

# 显著性检验
p_values = results.pvalues[1:]  # 忽略截距项
alpha = 0.05  # 显著性水平

# 根据显著性水平判断自变量的显著性
significant_vars = [var for var, p_value in p_values.items() if p_value < alpha]

# 输出显著的自变量
print("显著的自变量：", significant_vars)
