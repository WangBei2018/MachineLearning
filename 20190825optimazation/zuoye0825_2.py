from sympy import *
import numpy as np
from mpl_toolkits.mplot3d  import Axes3D
from matplotlib import pyplot as plt

fig = plt.figure()
ax = Axes3D(fig)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def fun(x,y):
    return (x-1)**4+y**2
def funx(x):
    return 4*(x-1)**3
def funy(y):
    return 2*y


x0 = 0;y0 = 0
x1 = 0;y1 = 1
xx = []
yy = []
ff = []
xx.append(0)
yy.append(1)
ff.append(2)
x0 = 0
y0 = 0
x1 = 0
y1 = 1
x,y = symbols('x y')
f = (x-1)**4+y**2
k = 1
cx = 1
cy = 1
while(np.absolute(cx) > 0.001 or np.absolute(cy) > 0.001 ):
    x0 = x1
    y0 = y1
    fx = diff(f,x)
    fy = diff(f,y)
    if (funx(x0) != 0):
        x1 = x0 - fun(x0,y0)/funx(x0)
    else:
        x1 = x0
    if(funy(y0) != 0):
        y1 = y0 - fun(x0,y0)/funy(y0)
    else:
        y1 = y0
    xx.append(x1)
    yy.append(y1)
    ff.append(f.subs({x:x1,y:y1}))
    k += 1
    cx = x1-x0
    cy = y1-y0

print('最终坐标点：(',x1,',',y1,')\n海拔：',f.subs({x:x1,y:y1}))
print('迭代次数：',k)



X = np.arange(-2,4,0.1)
Y = np.arange(-2,4,0.1)
X, Y = np.meshgrid(X, Y)
Z = (X-1)**4 + Y**2
ax.plot_surface(X, Y, Z)


ax.set_zlabel('Z/海拔')
ax.set_ylabel('Y')
ax.set_xlabel('X')
plt.title('山体走势函数图像')
plt.show()

plt.figure()
plt.contour(X, Y, Z)
plt.plot(xx,yy,'o-',markersize=2)
plt.ylabel('Y')
plt.xlabel('X')
plt.title('行走最优路线')
plt.text(0,1.1,'起始点')
plt.text(1,-0.3,'终点')
plt.show()

X1 = np.arange(0,20,1)
Y1 = ff
plt.plot(X1,Y1,'o-',markersize=2)
plt.title('海拔高度变化')
plt.ylabel('Y/海拔高度')
plt.xlabel('X/迭代次数')
plt.show()
