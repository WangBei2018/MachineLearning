import pandas as pd
import numpy as np
import time
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from itertools import cycle


def acc():
    numCorrect = [0] * 10
    numSum = [0] * 10
    predict_accuracy = [0] * 10
    for i in range(len(test_labels)):
        for j in range(10):
            if test_labels[i] == j:
                numSum[j] += 1
                if test_predict[i] == j:
                    numCorrect[j] += 1
    for j in range(10):
        predict_accuracy[j] = numCorrect[j] / numSum[j]

    return predict_accuracy


def drawROC():

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    y_predict = label_binarize(test_predict, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y_test = label_binarize(test_labels, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(10):
        fpr[i], tpr[i], threshold = roc_curve(y_test[:, i], y_predict[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        print(y_test[:, i], '\n', y_predict[:, i])
        print(fpr[i], '\n', tpr[i])

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'blue', 'green', 'orange', 'grey', 'black', 'pink'])
    for i, color in zip(range(10), colors):
        plt.plot(fpr[i], tpr[i], color=color, label='第{0}类ROC曲线 (面积 = {1:0.2f})'.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1])
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正确率')
    plt.ylabel('真正确率')
    plt.legend(loc="lower right")           #设置图例位置右下角
    plt.title('原始未降维数据')
    plt.show()



if __name__ == '__main__':

    # print("Start read data...")
    # time_1 = time.time()
    raw_data = pd.read_csv('2-mnist.csv', header=0)  # 读取csv数据
    data = raw_data.values
    features = data[::, 1::]
    labels = data[::, 0]


    # 随机选取30%数据作为测试集，剩余为训练集
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.3, random_state=0)

    # time_2 = time.time()
    # print('read data cost %f seconds' % (time_2 - time_1))

    # print('Start training...')
    clf = MultinomialNB(alpha=1.0) # 加入laplace平滑
    clf.fit(train_features, train_labels)
    # time_3 = time.time()
    # print('training cost %f seconds' % (time_3 - time_2))


    # print('Start predicting...')

    test_predict = clf.predict(test_features)
    # time_4 = time.time()
    # print('predicting cost %f seconds' % (time_4 - time_3))


    score = accuracy_score(test_labels, test_predict)
    print("准确率：%f" % score)
    # acc()
    drawROC()