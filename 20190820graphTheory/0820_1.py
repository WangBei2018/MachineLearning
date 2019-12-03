S = [0]
U = [1,2,3,4]
path = ['0','0','0','0','0']
inf = 999
zero = 100
length = [[zero,5,2,6,inf],[inf,zero,inf,inf,1],[inf,1,zero,3,5],[inf,inf,inf,zero,2],[inf,inf,inf,inf,zero]]

lmin = [0,0,0,0,0]       #当前最短路径
lmid = [0,0,0,0,0]       #中间更新路径
#print(length)
path[0] = '0->0'
lmin[0] = 0
for i in range(5):
    lmid[i] = length[0][i]

for i in range(4):
    mid = lmid.index(min(lmid))                 #找到中间更新路径中最短的那个的下标
    print('最小路径长度对应点序号:',mid)
    print('当前最短路径长度:          ',min(lmid))
    lmin[mid] = min(lmid)
    print('最短路径长度:             ',lmin)
    S.append(mid)                               #将找到的目前最小的路径的顶点加入S集合
    U.remove(mid)                               #将加入S的元素从U集合中删除
    print('添加后的S：               ',S)
    print('删除后的U：               ',U)
    if(i==0):
        path[mid] = path[mid] + '->' + str(mid) #将目前最短路径字符串表更新
    print('当前最短路径：            ',path)
    for j in range(5):                          #更新中间路径
        if(j != mid):
            t = lmid[mid] + length [mid][j]
            if(t < lmid[j]):
                lmid[j] = t
                path[j] = path[mid] + '->' + str(j)
#   print('更新后的lmid',i+1,':',lmid)
    lmid[lmid.index(min(lmid))] = 999           #将找到的最小路径从中间路径中删除防止下次继续选到，赋予一个较大的数

