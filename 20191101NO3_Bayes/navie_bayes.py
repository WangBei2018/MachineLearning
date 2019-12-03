import xlrd
import random
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def readExcel():
    data = xlrd.open_workbook('1-iris.xlsx')
    table = data.sheets()[0]
    y = table.col_values(4)
    return table


def normalization(table):
    data = []
    for i in range(4):
        data.append(table.col_values(i))
    for i in range(len(data)):
        data[i].remove(data[i][0])
        # print(data[i])
    maxlist = []
    minlist = []
    for i in range(len(data)):
        minlist.append(min(data[i]))
        maxlist.append(max(data[i]))
    # print('minlist:', minlist, '\nmaxlist:', maxlist)
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = (data[i][j]-minlist[i])/(maxlist[i]-minlist[i])
        # print(maxlist[i], minlist[i])
        # print(data[i])
    cl = table.col_values(4)
    #将类别添加到data中
    cl.remove(cl[0])
    data.append(cl)
    #返回的data为已经归一化处理后的数据
    return data


# 在原数据中选择三种类型数据各5条作为测试集，其余全为训练集
# 其中train一开始等于原始数据通过删除随机选择的下标删除元数据集中的15条数据
def selecttrain(ro):

    # print()
    train = ro
    test = []
    init = 50
    for i in range(5):
        index = random.randint(0, init-1)
        for j in range(3):
            test.append(train[init * j + index])
            # print(train[j][index])
        for j in range(3):
            train.remove(train[init * j + index - j])
        init -= 1
    # print('test:', test)
    # print('train:', train)
    return train, test


def predict(train, data):
    # print(data)
    # if lap == 1:
    #     count = np.ones(3, 4)
    # else:
    #     count = np.zeros((3, 4))
    count = np.zeros((3, 4))
    for i in range(len(train)):
        for j in range(4):
            # print('train', i, ':', train[i][j], 'data:', data[j], '类别：', train[i][4])
            if train[i][j] == data[j] and train[i][4] == 0:
                count[0][j] += 1
            elif train[i][j] == data[j] and train[i][4] == 1:
                count[1][j] += 1
            elif train[i][j] == data[j] and train[i][4] == 2:
                count[2][j] += 1
            else:
                continue
    for i in range(3):
        for j in range(4):
            # if lap == 1:
            #     count[i][j] = count[i][j] / 54
            # else:
            #     count[i][j] = count[i][j] / 50
            count[i][j] = count[i][j] / 50
    # print('count:', count)
    # result 的列表示三种分类，行表示分别使用1，2，3，4个特征对测试集进行分类时的概率
    result = np.ones((4, 3))
    for k in range(3):
        for l in range(4):
            for j in range(l+1):
                # print(result[k][j], 'before')
                result[l][k] = result[l][k] * count[k][j]
    # print('result:', result)
    classes = []
    for i in range(4):
        classes.append(list(result[i]).index(max(result[i])))
    # print(classes)
    return classes, result


def laplace_predict(train, data):
    # print(data)
    count = np.ones((3, 4))
    for i in range(len(train)):
        for j in range(4):
            # print('train', i, ':', train[i][j], 'data:', data[j], '类别：', train[i][4])
            if train[i][j] == data[j] and train[i][4] == 0:
                count[0][j] += 1
            elif train[i][j] == data[j] and train[i][4] == 1:
                count[1][j] += 1
            elif train[i][j] == data[j] and train[i][4] == 2:
                count[2][j] += 1
            else:
                continue
    for i in range(3):
        for j in range(4):
            count[i][j] = count[i][j] / 54
    # print('count:', count)
    # result 的列表示三种分类，行表示分别使用1，2，3，4个特征对测试集进行分类时的概率
    result = np.ones((4, 3))
    for k in range(3):
        for l in range(4):
            for j in range(l + 1):
                # print(result[k][j], 'before')
                result[l][k] = result[l][k] * count[k][j]
    # print('result:', result)
    classes = []
    for i in range(4):
        classes.append(list(result[i]).index(max(result[i])))
    # print(classes)
    return classes, result


def bayes(data):
    # col = []
    # # cl分类存放数据cl[0~3]为类别1的四个特征 4~7类别二的四个特征 8~11类别三的四个特征
    # for i in range(3):
    #     for j in range(5):
    #         col.append(data[j][i * 50:(i + 1) * 50])

    # ro存放150条数据，每条数据为Excel表格中的一行
    ro = []
    for i in range(150):
        a = []
        for j in range(5):
           a.append(data[j][i])
        ro.append(a)
    [train, test] = selecttrain(ro)
    # predict函数计算测试集分类结果，返回值为预测结果以及预测概率矩阵
    y = []
    prob = []
    plt.figure(figsize=(10, 10))
    # 未平滑绘制ROC曲线
    for k in range(4):
        plt.figure()
        for j in range(3):
            for i in range(len(test)):
                [classes, resultmatrix] = predict(train, test[i])
                # print('测试集类别：', test[i][4], '\n预测类别：', classes)
                if int(test[i][4]) == j:
                    y.append(0)
                else:
                    y.append(1)
                if classes[k] == j:
                    prob.append(0)
                else:
                    prob.append(1)
            # print(y, '\n', prob)
            fpr, tpr, threshold = roc_curve(y, prob)
            roc_auc = auc(fpr, tpr)
            if j == 0:
                col = 'darkorange'
            elif j == 1:
                col = 'g'
            else:
                col = 'y'
            plt.plot(fpr, tpr, color=col,
                     lw=2, label='ROC 曲线 (面积 = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('假正确率')
            plt.ylabel('真正确率')
            plt.title(str(k+1)+'个特征分类结果ROC曲线')
            plt.legend(loc="lower right")
    plt.show()

    # 拉普拉斯平滑后ROC曲线
    for k in range(4):
        plt.figure()
        for j in range(3):
            for i in range(len(test)):
                [classes, resultmatrix] = laplace_predict(train, test[i])
                # print('测试集类别：', test[i][4], '\n预测类别：', classes)
                if int(test[i][4]) == j:
                    y.append(0)
                else:
                    y.append(1)
                if classes[k] == j:
                    prob.append(0)
                else:
                    prob.append(1)
            # print(y, '\n', prob)
            fpr, tpr, threshold = roc_curve(y, prob)
            roc_auc = auc(fpr, tpr)
            if j == 0:
                col = 'darkorange'
            elif j == 1:
                col = 'g'
            else:
                col = 'y'
            plt.plot(fpr, tpr, color=col,
                     lw=2, label='ROC 曲线 (面积 = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('假正确率')
            plt.ylabel('真正确率')
            plt.title(str(k+1)+'个特征分类结果ROC曲线(拉普拉斯平滑后)')
            plt.legend(loc="lower right")
    plt.show()

    return 0


if __name__ == '__main__':
    table = readExcel()
    data = normalization(table)
    # lap = input("是否平滑(1：是 0：否)  : ")
    bayes(data)
