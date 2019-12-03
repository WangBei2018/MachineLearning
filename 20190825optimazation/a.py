import numpy as np


# 二维原始图像
def fun(x, y):
    return (x-1)**4+y**2
## 偏函数
def hx(x, y):
    return 3*(x-1)**3
def hy(x, y):
    return 2*y

x = 0;y = 1;alpha = 0.5
# 定义y的变化量和迭代次数
y_change = fun(x, y)
change = 1
iter_num = 0

while (change > 1e-5 and iter_num < 10000):
    tmp_x = x - alpha * hx(x, y)
    tmp_y = y - alpha * hy(x, y)

    #tmp_z = fun(tmp_x, tmp_y)
    #y_change = np.absolute(tmp_z - fun(x,y))
    change = np.absolute(tmp_x-x)
    x = tmp_x
    y = tmp_y
    iter_num += 1
print(u"最终结果为:(%.5f, %.5f, %.5f)" % (x,y, fun(x,y)))
print(u"迭代过程中X的取值，迭代次数:%d" % iter_num)
