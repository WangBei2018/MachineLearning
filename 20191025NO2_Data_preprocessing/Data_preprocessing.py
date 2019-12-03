import matplotlib.pyplot as plt
import numpy as np
import xlrd
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def readExcel():
    data = xlrd.open_workbook('data.xls')
    table = data.sheets()[0]
    y = table.col_values(0)
    x1 = table.col_values(1)
    x2 = table.col_values(2)
    return y, x1, x2


def normalization(x1, x2):
    min1 = min(x1)
    max1 = max(x1)
    min2 = min(x2)
    max2 = max(x2)
    for i in range(len(x1)):
        x1[i] = (x1[i] - min1) / (max1 - min1)
        x2[i] = (x2[i] - min2) / (max2 - min2)
    return x1, x2


def logical_regrassion(x1, x2, y):
    a = 0.2
    p0 = 1
    p1 = 1
    p2 = 1
    # print(y)
    for j in range(1000):  # 迭代1000次
        sum1 = 0
        sum2 = 0
        sum0 = 0
        for i in range(len(y)):
            sum0 = sum0 + ((1 + np.e ** (-(p0 + p1 * x1[i] + p2 * x2[i]))) ** (-1) - y[i])
            sum1 = sum1 + ((1 + np.e ** (-(p0 + p1 * x1[i] + p2 * x2[i]))) ** (-1) - y[i]) * x1[i]
            sum2 = sum2 + ((1 + np.e ** (-(p0 + p1 * x1[i] + p2 * x2[i]))) ** (-1) - y[i]) * x2[i]
        p0 = p0 - a * sum0
        p1 = p1 - a * sum1
        p2 = p2 - a * sum2

    # print(p0, p1, p2)
    return p0, p1, p2


def measure(p0, p1, p2, x1, x2, y):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(y)):
        if (p0 + p1 * x1[i] + p2 * x2[i]) > 0 and y[i] == 1:
            TP += 1
        elif (p0 + p1 * x1[i] + p2 * x2[i]) < 0 and y[i] == 1:
            FN += 1
        elif (p0 + p1 * x1[i] + p2 * x2[i]) > 0 and y[i] == 0:
            FP += 1
        else:
            TN += 1
    # print('TP:',TP,'\nTN:', TN,'\nFP:', FP,'\nFN:', FN)
    ACC = (TP + TN)/(TP + TN + FP + FN)
    TPR = TP/(TP + FN)
    TNR = TN/(TN + FP)
    g_mean = (TPR*TNR) ** (1/2)
    P = TP/(TP + FP)
    R = TPR
    F = 2 * P * R / (P + R)
    return g_mean, ACC, F


def smote(x1, x2, y):            #SMOTE过采样
    smo = SMOTE(random_state=10)
    xloc = []
    for i in range(1100):
        xloc.append((x1[i], x2[i]))
    # print('xloc:', xloc)
    X_smo, y_smo = smo.fit_sample(xloc, y)
    # print('X_smo', X_smo, '\ny_smo', y_smo)
    for i in range(len(y_smo)):
        if y_smo[i] == 1:
            plt.scatter(X_smo[i, 0], X_smo[i, 1], c='r', s=10)
        else:
            plt.scatter(X_smo[i, 0], X_smo[i, 1], c='b', s=10)
    [psmo0, psmo1, psmo2] = logical_regrassion(X_smo[:, 0], X_smo[:, 1], y_smo)
    [g_mean_smo, ACC_smo, F_smo] = measure(psmo0, psmo1, psmo2, X_smo[:, 0], X_smo[:, 1], y_smo)
    Y_smo = (psmo0 + psmo1 * X) / (-psmo2)  # p0+p1*x1+p2*x2=0
    plt.plot(X, Y_smo)
    plt.title("SMOTE过采样逻辑回归")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.text(0.2, 1.2, str(round(psmo1, 2)) + 'x1' + '+' + str(round(psmo2, 2)) + 'x2' + str(round(psmo0, 2)) + ' = 0')
    plt.text(0, -0.5, 'G-mean:' + str(g_mean_smo) + '\nF-measure:' + str(F_smo))
    plt.show()
    return


def k_means(x1, x2):          #K-means聚类欠采样
    kms = KMeans(n_clusters=100)
    x = []
    for i in range(1000):
        x.append([x1[100 + i], x2[100 + i]])
    # print(x)
    cluster = kms.fit_predict(x)
    center = kms.cluster_centers_
    xkmeans1 = []
    xkmeans2 = []
    ykmeans = []
    for i in range(100):
        xkmeans1.append(x1[i])
        xkmeans2.append(x2[i])
    for i in range(100):
        xkmeans1.append(center[i, 0])
        xkmeans2.append(center[i, 1])
    for i in range(100):
        ykmeans.append(0)
    for i in range(100):
        ykmeans.append(1)
    # print('xkmeans1:',xkmeans1, '\nxkmeans2:', xkmeans2, '\nymeans:', ykmeans)
    [pkm0, pkm1, pkm2] = logical_regrassion(xkmeans1, xkmeans2, ykmeans)
    # print(pkm0, pkm1, pkm2)

    for i in range(len(ykmeans)):
        if y[i] == 1:
            plt.scatter(xkmeans1[i], xkmeans2[i], c='r', s = 10)
        else:
            plt.scatter(xkmeans1[i], xkmeans2[i], c='b', s = 10)
    [g_mean_km, ACC_km, F_km] = measure(pkm0, pkm1, pkm2, xkmeans1, xkmeans2, ykmeans)
    Ykm = (pkm0 + pkm1 * X) / (-pkm2)  # p0+p1*x1+p2*x2=0
    plt.plot(X, Y)
    plt.title("K-Means聚类欠采样后逻辑回归")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.text(0.2, 1.5, str(round(pkm1, 2)) + 'x1' + '+' + str(round(pkm2, 2)) + 'x2' + str(round(pkm0, 2)) + ' = 0')
    plt.text(0, -0.5, 'G-mean:' + str(g_mean_km) + '\nF-measure:' + str(F_km))
    plt.show()

    return


if __name__ == '__main__':

    [y, x1, x2] = readExcel()
    [x1, x2] = normalization(x1, x2)
    for i in range(len(y)):
        y[i] -= 1
    # print(x1, '\n', x2)
    for i in range(len(y)):
        if y[i] == 1:
            plt.scatter(x1[i], x2[i], c='r', s=10)
        else:
            plt.scatter(x1[i], x2[i], c='b', s=10)
    # plt.show()
    [p0, p1, p2] = logical_regrassion(x1, x2, y)

    # 使用评价指标G-mean
    [g_mean, ACC, F] = measure(p0, p1, p2, x1, x2, y)
    X = np.linspace(0, 1, 1000)
    Y = (p0 + p1 * X) / (-p2)  # p0+p1*x1+p2*x2=0
    plt.plot(X, Y)
    plt.title("原始数据逻辑回归")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.text(0.2, 1.5, str(round(p1, 2)) + 'x1' + '+' + str(round(p2, 2)) + 'x2' + str(round(p0, 2)) + ' = 0')
    plt.text(0, -0.5, 'G-mean:' + str(g_mean) + '\nF-measure:' + str(F))
    plt.show()
    # K-Means聚类欠采样
    k_means(x1, x2)
    # SMOTE过采样
    smote(x1, x2, y)
