import random
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
x1=np.zeros(100)
x2=np.zeros(100)
y=np.zeros(100)
for i in range(100):
    x1[i] = random.random()
    x2[i] = x1[i] ** 2
    y[i] = 20 + 5 * x1[i] + 5 * x2[i] + random.random() * 2-1

plt.scatter(x1, y)#散点图

a = 0.2
p0 = 1
p1 = 1
p2 = 1
m = 100
for j in range(10000):
    sum1 = 0
    sum2 = 0
    sum0 = 0
    for i in range(100):
        sum0 = sum0 + (p0 + p1 * x1[i] + p2 * x2[i] - y[i])
        sum1 = sum1 + (p0 + p1 * x1[i] + p2 * x2[i] - y[i]) * x1[i]
        sum2 = sum2 + (p0 + p1 * x1[i] + p2 * x2[i] - y[i]) * x2[i]
    p0 = p0 - a * (1/m) * sum0
    p1 = p1 - a * (1/m) * sum1
    p2 = p2 - a * (1/m) * sum2
print(p0, p1, p2)

X = np.linspace(0, 1, 1000)
Y = p0 + p1 * X + p2 * X**2
plt.plot(X, Y)
plt.title('散点及拟合曲线')
plt.text(0, 29, 'h(x) = '+str(round(p0, 2))+' + '+str(round(p1, 2))+'x '+' + '+str(round(p2, 2))+'x^2')
plt.xlabel('X')
plt.ylabel('h(X)')
plt.show()