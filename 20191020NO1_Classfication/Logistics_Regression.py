import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

y=[]
A = [[np.random.uniform(0,0.5) for i in range(100)],[np.random.uniform(0,0.5) for i in range(100)]]
#B的横纵坐标都在(8,9)之间
for i in range(100):
    y.append(0)
B = [[np.random.uniform(0.5,1) for i in range(100)],[np.random.uniform(0.5,1) for i in range(100)]]
for i in range(100):
    y.append(1)

x1 = A[0] + B[0]
x2 = A[1] + B[1]
print(len(x1))


plt.scatter(A[0],A[1],c='b')
plt.scatter(B[0],B[1],c='r')
#plt.show()

a = 0.2
p0 = 1
p1 = 1
p2 = 1
for j in range(1000):  # 迭代1000次
    sum1 = 0
    sum2 = 0
    sum0 = 0
    for i in range(200):
        sum0 = sum0 + ((1 + np.e ** (-(p0 + p1 * x1[i] + p2 * x2[i]))) ** (-1) - y[i])
        sum1 = sum1 + ((1 + np.e ** (-(p0 + p1 * x1[i] + p2 * x2[i]))) ** (-1) - y[i]) * x1[i]
        sum2 = sum2 + ((1 + np.e ** (-(p0 + p1 * x1[i] + p2 * x2[i]))) ** (-1) - y[i]) * x2[i]
    p0 = p0 - a * sum0
    p1 = p1 - a * sum1
    p2 = p2 - a * sum2

print(p0, p1, p2)
X = np.linspace(0, 1, 1000)
Y = (p0 + p1 * X) / (-p2)  # p0+p1*x1+p2*x2=0
plt.title('逻辑回归')
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(X, Y)
plt.text(0.1,0.9,str(round(p1,2))+'x1'+'+'+str(round(p2,2))+'x2'+str(round(p0,2))+' = 0')
plt.show()