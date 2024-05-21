import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from scipy.stats import f

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

# 使用线性回归分析进攻效率对胜负的影响
X = df['进攻效率']  # 自变量：进攻效率
Y = df['胜负']     # 因变量：胜负

# 添加截距项
X = sm.add_constant(X)

# 拟合线性回归模型
model = sm.OLS(Y, X)
results = model.fit()

# 输出回归结果
print(results.summary())

# 绘制散点图和拟合直线
plt.scatter(X['进攻效率'], Y, label='Actual Data')
plt.plot(X['进攻效率'], results.fittedvalues, color='red', label='Fitted Line')
plt.xlabel('进攻效率')
plt.ylabel('胜负')
plt.title('进攻效率对比赛胜负的影响')
plt.legend()
plt.show()

# 显著性检验结果
f_statistic = results.fvalue
p_value = results.f_pvalue

# 计算临界值和p值
alpha = 0.1  # 置信水平为90%
dfn, dfd = 1, df.shape[0] - 2  # 自由度
f_critical = f.ppf(1 - alpha, dfn, dfd)
p_value_90 = 1 - f.cdf(f_statistic, dfn, dfd)

alpha = 0.05  # 置信水平为95%
f_critical = f.ppf(1 - alpha, dfn, dfd)
p_value_95 = 1 - f.cdf(f_statistic, dfn, dfd)

# 输出结论
print("F值：", f_statistic)
print("p值 (90%置信水平)：", p_value_90)
print("p值 (95%置信水平)：", p_value_95)
