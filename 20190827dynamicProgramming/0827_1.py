from scipy.sparse import coo_matrix
import numpy as np
from collections import Counter
from itertools import chain
row = [0,0,0,0,1,1,1,2,2,2,3,3,3,4,4,5,5,6,6,7,8,9]
col = [0,1,2,3,4,5,6,4,5,6,4,5,6,7,8,7,8,7,8,9,9,9]
data = [0,2,5,1,12,14,10,6,10,4,13,11,12,3,9,6,5,8,10,5,2,0]
dis = coo_matrix((data,(row,col)),shape=(10,10))
name = [['A'],['B1','B2','B3'],['C1','C2','C3'],['D1','D2'],['E']]
D = dis.todense()
num = 0
for i in range(0,len(name)):
    num = num + len(name[i])
f = np.zeros((num,1))          #最短路径长度
loc = []
f[9] = 0
x = []        #最短路径
numE = 9
N = list(chain(*name))                                  #将顶点名称拉成一行
print('\n')
for k in range(len(name)-1,0,-1):
    numS = numE - len(name[k-1])                        #当前起点所在位置编号
    for j in range(len(name[k-1])-1,-1,-1):
        d = []
        for i in range(0,len(name[k])):
            d.append(D[numS+j,numE+i]+f[numE+i])        #计算第K个阶段，第numS+j个顶点到下一个阶段顶点之间的距离
        f[numS+j] = min(d)                              #求出距离中的最小值
        loc.append(d.index(min(d))+numE)                #记录最小值所在顶点的编号
        print(N[numS+j],':',list(chain(*d)) )
        print('f(',N[numS+j],')=',min(d),'\n')
    numE = numS                                         #更新当前中间点

for i in range(0,len(loc)):
    x.append(N[loc[i]])                                 #将每个节点的决策存入x
    print('X(',N[len(loc)-1-i],')=',N[loc[i]])
#print(x)
t = loc[len(loc)-1]
route = []
#route.append('A')
for i in range(0,len(name)-1):
    route.append(N[t])                                  #route为路径名称
    t = loc[len(loc)-1-t]
#print(route)
print('最短路径：  A',end='')
for x in route:
    print('->',x,end='')                                #输出最终路线
