import matplotlib.pyplot as plt
import xlrd
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
def read_excel():
    file = xlrd.open_workbook('data.xlsx')
    data = file.sheets()[0]
    x0 = data.col_values(0)
    x1 = data.col_values(1)
    x2 = data.col_values(2)
    return x0, x1, x2


def show(x0, x1, x2):

    ax = plt.figure().add_subplot(111, projection='3d')
    ax.scatter(x0, x1, x2, s=5)
    plt.title("原始数据三维散点图")
    plt.show()
    return


def decentralized(x0, x1, x2):
    mean0 = np.mean(x0)
    mean1 = np.mean(x1)
    mean2 = np.mean(x2)
    # print(mean0, '\n', mean1, '\n', mean2)
    for i in range(len(x0)):
        x0[i] -= mean0
        x1[i] -= mean1
        x2[i] -= mean2
    return x0, x1, x2


def pca(x0, x1, x2):
    [x0, x1, x2] = decentralized(x0, x1, x2)                                #去中心化
    covmatrix = np.dot(np.array([x0, x1, x2]), np.array([x0, x1, x2]).T)    #求协方差矩阵
    # print(covmatrix)
    [lamda,vec] = np.linalg.eig(covmatrix)                                  #求矩阵特征值特征向量
    eigenvector = []                                                        #前两维协方差矩阵
    for i in range(len(vec)):                                               #取特征值最大的前两维特征向量
        if i != list(lamda).index(min(list(lamda))):
            eigenvector.append(list(vec[i]))
    eigenvector = np.array(eigenvector).T
    m = np.dot(np.array([x0, x1, x2]).T, eigenvector)
    m = m.T
    # X = list(m[0])
    # Y = list(m[1])
    # print(m[0])
    # # plt.scatter(X, Y)
    # for i in range(len(m[0])):
    #     plt.scatter(X[i], Y[i])
    plt.scatter(m[0], m[1])
    plt.title("降成2维后的散点图")
    plt.show()
    return


if __name__ == '__main__':
    [x0, x1, x2] = read_excel()
    show(x0, x1, x2)
    pca(x0, x1, x2)