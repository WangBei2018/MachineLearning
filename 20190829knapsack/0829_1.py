def bag(n, c, w, v):
    # 置零，表示初始状态
    value = [[0 for j in range(c + 1)] for i in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, c + 1): #其中i表示物品的数量，而j表示最大的重量限制。
            value[i][j] = value[i - 1][j]
            # 背包总容量够放当前物体，遍历前一个状态考虑是否置换
            if j >= w[i - 1] and value[i][j] < value[i - 1][j - w[i - 1]] + v[i - 1]:

                value[i][j] = value[i - 1][j - w[i - 1]] + v[i - 1]
            print('当物品数量为', i,'时,物品最大重量限制为', j,'背包价值为:',value[i][j])

    value1 = [[0 for j in range(c)] for i in range(n)]
    for i in range(1,len(value)):
        for j in range(1,len(value[i])):
            value1[len(value)-i-1][j-1] = value[i][j]
    # for i in range(1,len(value)):
    #         print(value[i])
    for x in value1:
        print(x)
    return value


def show(n, c, w, value):
    print('最大价值为:', value[n][c])
    x = [False for i in range(n)]
    j = c
    for i in range(n, 0, -1):
        if value[i][j] > value[i - 1][j]:
            x[i - 1] = True
            j -= w[i - 1]
    print('背包中所装物品为:')
    for i in range(n):
        if x[i]:
            print('第', i+1, '个,', end='')


n = 5  #物品的数量，
c = 10 #书包能承受的重量，
w = [4,5,6,2,2] #每个物品的重量，
v = [6,4,5,3,6] #每个物品的价值
value = bag(n,c,w,v)
show(n,c,w,value)
#函数c[i, w]表示到第i个元素为止，在限制总重量为w的情况下我们所能选择到的最优解。
