# 状态0，5 =( M,W,G,C ); 状态1，6 =(M,W,G);
# 状态2，7 =(M,W,C); 状态3，8=(M,G,C); 状态4，9 =(M,G)
S = [0]
U = [1,2,3,4,5,6,7,8,9]
inf = 999
path = ['0','0','0','0','0','0','0','0','0','0']
length = [[inf,inf,inf,inf,inf,inf,inf,inf,inf,1],
          [inf,inf,inf,inf,inf,inf,inf,1,1,inf],
          [inf,inf,inf,inf,inf,inf,1,inf,1,1],
          [inf,inf,inf,inf,inf,inf,1,1,inf,inf],
          [inf,inf,inf,inf,inf,1,inf,1,inf,inf],
          [inf,inf,inf,inf,1,inf,inf,inf,inf,inf],
          [inf,inf,1,1,inf,inf,inf,inf,inf,inf],
          [inf,1,inf,1,1,inf,inf,inf,inf,inf],
          [inf,1,1,inf,inf,inf,inf,inf,inf,inf],
          [1,inf,1,inf,inf,inf,inf,inf,inf,inf]
          ]
lmin = [0,0,0,0,0,0,0,0,0,0]       #当前最短路径
lmid = [0,0,0,0,0,0,0,0,0,0]       #中间更新路径
#print(length)
path[0] = '0->0'
lmin[0] = 0
for i in range(10):
    lmid[i] = length[0][i]

#print(lmid)
p = []
p.append(0)
for i in range(9):
    print('\n第',i+1,'次：')
    mid = lmid.index(min(lmid))                 #找到中间更新路径中最短的那个的下标
    print('最小路径长度对应点序号:    ',mid)
    p.append(mid)
    print('当前最短路径长度:          ',min(lmid))
    lmin[mid] = min(lmid)
    print('最短路径长度:             ',lmin)
    S.append(mid)                               #将找到的目前最小的路径的顶点加入S集合
#    U.remove(mid)                               #将加入S的元素从U集合中删除
    print('添加后的S：               ',S)
#    print('删除后的U：               ',U)
    if(i==0):
        path[mid] = path[mid] + '->' + str(mid) #将目前最短路径字符串表更新
    print('当前最短路径：            ',path)
    for j in range(10):                          #更新中间路径
        if(j != mid and j not in(p)):
            t = lmid[mid] + length [mid][j]
            if(t < lmid[j]):
                lmid[j] = t
                path[j] = path[mid] + '->' + str(j)
#    print('更新后的lmid',i+1,':',lmid)
    lmid[lmid.index(min(lmid))] = inf           #将找到的最小路径从中间路径中删除防止下次继续选到，赋予一个较大的数


print('\n人狼羊问题的解：',path[5])