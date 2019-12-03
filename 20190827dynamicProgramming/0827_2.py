import numpy as np

g = [[0,3,7,9,12],
[0,5,10,11,11,11],
[0,4,6,11,12,12]]
f = np.zeros(3,5)
profit = []
for i in range(0,len(g)):
    for j in range(0,len(g[1])):
        for k in range(0,j):
            profit.append(g[i,k]+f[i,k])