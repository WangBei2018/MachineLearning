import xlrd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
def read_excel():
    file = xlrd.open_workbook('CORK STOPPERS.XLS')
    rawdata = file.sheets()[1]
    x = []
    y = rawdata.col_values(1)
    for i in range(12):
        x.append(rawdata.col_values(i))
    # y.remove(y[0])
    # y.remove(y[0])
    for i in range(12):
        t = x[i]
        t.remove(t[0])
        t.remove(t[0])
    x = np.array(x)
    data = x[2::, ::]
    y = x[1, ::]

    return data, y


def sbs(data, y):
    datatrain = data.T
    index = np.zeros(9)
    acc = []
    features = datatrain
    labels = y
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.3,
                                                                                random_state=10)
    clf = GaussianNB()  # 加入laplace平滑
    clf.fit(train_features, train_labels)
    test_predict = clf.predict(test_features)
    score = accuracy_score(test_labels, test_predict)
    acc.append(score)
    for i in range(len(data)-1):
        max = 0
        for j in range(len(data)-i):
            features = datatrain
            features = np.delete(features, j, 1)
            labels = y
            # print('features: ', features)
            # print('length:', len(features))
            train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.3,
                                                                                        random_state=10)
            clf = GaussianNB()  # 加入laplace平滑
            clf.fit(train_features, train_labels)
            test_predict = clf.predict(test_features)
            score = accuracy_score(test_labels, test_predict)
            # print('次数： ', j)
            # print("准确率：%f" % score)
            if max < score:
                max = score
                num = j
                index[i] = j
        acc.append(max)
        # print(num)
        datatrain = np.delete(datatrain, num, 1)

    feature_index = []
    for i in range(10):
        feature_index.append(i)
    sortindex = []
    for i in range(9):
        num = int(index[i])
        sortindex.append(feature_index[num])
        del feature_index[num]
    sortindex.append(feature_index[0])
    print('删除特征的顺序： ', sortindex)
    print(acc)
    x = range(len(sortindex))
    plt.plot(x, acc, marker='o', mec='r', mfc='w')
    plt.title("依次删除特征准确率变化折线图")
    plt.xlabel("删除的特征个数")
    plt.ylabel("分类准确率")
    plt.show()
    return


if __name__ == '__main__':
    [data, y] = read_excel()
    sbs(data, y)