# 逻辑回归


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

excelFile = 'iris.csv'
df = pd.DataFrame(pd.read_csv(excelFile))
data = df[[u'花萼长度', u'花萼宽度', u'class']]

a=data['花萼长度'].copy()
x1 = list(data['花萼长度'].copy())
x2 = list(data['花萼宽度'].copy())
y = list(data['class'].copy())

for i in range(len(x2)):
    if y[i] == 0:
        plt.scatter(x1[i], x2[i], c='r')
    if y[i] == 1:
        plt.scatter(x1[i], x2[i], c='g')
    else:
        # plt.scatter(x1[i], x2[i], c='b')
        print(y[i])

a = 0.2
p0 = 1
p1 = 1
p2 = 1
for j in range(1000):  # 迭代1000次
    sum1 = 0
    sum2 = 0
    sum0 = 0
    for i in range(100):
        sum0 = sum0 + ((1 + np.e ** (-(p0 + p1 * x1[i] + p2 * x2[i]))) ** (-1) - y[i])
        sum1 = sum1 + ((1 + np.e ** (-(p0 + p1 * x1[i] + p2 * x2[i]))) ** (-1) - y[i]) * x1[i]
        sum2 = sum2 + ((1 + np.e ** (-(p0 + p1 * x1[i] + p2 * x2[i]))) ** (-1) - y[i]) * x2[i]
    p0 = p0 - a * sum0
    p1 = p1 - a * sum1
    p2 = p2 - a * sum2

print(p0, p1, p2)

X = np.linspace(4, 8.5, 10000)
Y = (p0 + p1 * X) / (-p2)  # p0+p1*x1+p2*x2=0
plt.title('鸢尾花数据逻辑回归')
plt.xlabel('花萼长度')
plt.ylabel('花萼宽度')
plt.plot(X, Y)
plt.show()
