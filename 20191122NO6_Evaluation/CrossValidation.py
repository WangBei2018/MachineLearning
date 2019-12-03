import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import neighbors
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
import functools
import operator
import copy
from sklearn.metrics import accuracy_score, precision_recall_curve, classification_report, confusion_matrix, roc_curve, \
    auc, precision_score, recall_score

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def read_file():
    rawdata = pd.read_excel("../data/1-iris.xlsx")
    data = rawdata.values
    data = data[50:150, :]
    np.random.shuffle(data)         # 打乱数组
    labels = data[:, 4]
    labels = np.array([x - 1 for x in labels])
    features = data[:, 0:4]

    return features, labels


def decentralized(x):  # 去中心化
    mean = dict()
    for i in range(4):
        mean[i] = np.mean(x[:, i])
    for i in range(len(mean)):
        for j in range(len(x[:, i])):
            x[j][i] -= mean[i]

    return x


def pca(features, labels):
    # features = decentralized(features)                              #去中心化
    covmatrix = np.dot(np.array(features).T, np.array(features))  # 求协方差矩阵
    [lamda, vec] = np.linalg.eig(covmatrix)  # 求矩阵特征值特征向量

    # #####降维到三维空间#####
    # eigenvector = []
    # for i in range(len(vec)):  # 取特征值最大的前两维特征向量
    #     if i != list(lamda).index(min(list(lamda))):
    #         eigenvector.append(list(vec[i]))
    # eigenvector = np.array(eigenvector).T
    # m = np.dot(np.array(features), eigenvector)
    # # print(m)
    # ax = plt.figure().add_subplot(111, projection='3d')
    # # ax.scatter(m[:, 0], m[:, 1], m[:, 2], s=15)
    # for i in range(len(labels)):
    #     if labels[i] == 0:
    #         ax.scatter(m[i, 0], m[i, 1], m[i, 2], s=15, c='r')
    #     else:
    #         ax.scatter(m[i, 0], m[i, 1], m[i, 2], s=15, c='g')
    #
    # plt.title("三维散点图")
    # plt.show()

    ####降维到二维空间####
    eigenvector = vec
    for i in range(2):
        for j in range(len(eigenvector[i])):
            if j == list(lamda).index(min(list(lamda))):
                eigenvector = np.delete(eigenvector, j, axis=1)
                lamda = np.delete(lamda, j)
                # eigenvector = np.array(eigenvector).T
                # print(eigenvector)
    m = np.dot(np.array(features), eigenvector)
    # print(m)

    for i in range(len(labels)):
        if labels[i] == 0:
            plt.scatter(m[i, 0], m[i, 1], s=15, c='r')
        else:
            plt.scatter(m[i, 0], m[i, 1], s=15, c='g')
    plt.title("二维散点图")
    # plt.show()

    return m


def functools_reduce(a):
    return functools.reduce(operator.concat, a)


def cross_validation(k, features, labels):
    k_features = np.split(features, k, axis=0)
    k_labels = np.split(labels, k, axis=0)
    return k_features, k_labels


def bayes(k_features, k_labels):
    # print('k_features:', k_features)
    clf = GaussianNB()
    y_pred = dict()
    for j in range(len(k_labels)):
        k_features[j] = k_features[j].tolist()
        k_labels[j] = k_labels[j].tolist()
    scores = []
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    plt.figure()
    for k in range(len(k_labels)):
        k_fea = copy.deepcopy(k_features)
        k_lab = copy.deepcopy(k_labels)
        k_fea.remove(k_fea[k])
        k_lab.remove(k_lab[k])
        k_fea = functools_reduce(k_fea)
        k_lab = functools_reduce(k_lab)
        clf = clf.fit(k_fea, k_lab)

        y_pred[k] = clf.predict(k_features[k])
        score = clf.predict_proba(k_features[k])
        #         # score = score[:, 1].tolist()
        #         # scores.append(score)
        fpr, tpr, thresholds = roc_curve(np.array(k_labels[k]), score[:, 1])
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        # plt.plot(fpr, tpr, lw=2, alpha=0.3,
        #          label='ROC curve %d (area = %0.2f)' % (k, roc_auc))



        precision, recall, thresholds = precision_recall_curve(k_labels[k], y_pred[k])
        plt.plot(precision, recall, label=r'P-R curve %d' % k)
        # print('precision: ', precision, '\nrecall: ', recall)
        # plt.title("P-R曲线")
        # plt.plot(precision, recall)
        # # conf_mat = confusion_matrix(k_labels[k], labels=labels)       #混淆矩阵
        # # acc = accuracy_score(k_labels[k], y_pred[k])                  #准确率

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    # plt.plot(mean_fpr, mean_tpr, c='r',
    #          label=r'Mean ROC (area = %0.2f)' % (mean_auc),
    #          lw=2, alpha=.8)
    # plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    # plt.title("朴素贝叶斯5折交叉验证后ROC曲线")
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.legend(loc="lower right")

    plt.title("朴素贝叶斯分类器P-R曲线")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="lower left")
    plt.xlim([0, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.show()
    plt.show()
    return


def knn_classifier(k_features, k_labels):

    knn = neighbors.KNeighborsClassifier()
    y_pred = dict()
    # for j in range(len(k_labels)):
    #     k_features[j] = k_features[j].tolist()
    #     k_labels[j] = k_labels[j].tolist()
    scores = []
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    plt.figure()
    pre = []
    rec = []
    for k in range(len(k_labels)):
        k_fea = copy.deepcopy(k_features)
        k_lab = copy.deepcopy(k_labels)
        k_fea.remove(k_fea[k])
        k_lab.remove(k_lab[k])
        k_fea = functools_reduce(k_fea)
        k_lab = functools_reduce(k_lab)
        knn = knn.fit(k_fea, k_lab)

        y_pred[k] = knn.predict(k_features[k])
        # score = knn.predict_proba(k_features[k])
        # #         # score = score[:, 1].tolist()
        # #         # scores.append(score)
        # fpr, tpr, thresholds = roc_curve(np.array(k_labels[k]), score[:, 1])
        # tprs.append(np.interp(mean_fpr, fpr, tpr))
        # tprs[-1][0] = 0.0
        # roc_auc = auc(fpr, tpr)
        # aucs.append(roc_auc)
        # plt.plot(fpr, tpr, lw=2, alpha=0.3,
        #          label='ROC curve %d (area = %0.2f)' % (k, roc_auc))

        precision, recall, thresholds = precision_recall_curve(k_labels[k], y_pred[k])
        pre.append(precision_score(k_labels[k], y_pred[k]))
        rec.append(recall_score(k_labels[k], y_pred[k]))
        # pre.append(precision)
        # rec.append(recall)
        # print('precision: ', precision, '\nrecall: ', recall)
        plt.title("KNN分类器P-R曲线")
        plt.plot(precision, recall, label=r'P-R curve %d' % k)

    # mean_tpr = np.mean(tprs, axis=0)
    # mean_tpr[-1] = 1.0
    # mean_auc = auc(mean_fpr, mean_tpr)
    # std_auc = np.std(aucs)
    # plt.plot(mean_fpr, mean_tpr, c='r',
    #          label=r'Mean ROC (area = %0.2f)' % (mean_auc),
    #          lw=2, alpha=.8)
    # plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    # plt.title("KNN5折交叉验证后ROC曲线")
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    print(pre)
    print(rec)
    # mean_pre = np.mean(pre, axis=0)
    # mean_rec = np.mean(rec, axis=0)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="lower left")
    plt.xlim([0, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.show()

    return



if __name__ == '__main__':
    [features, labels] = read_file()
    features = pca(features, labels)
    # k = input("交叉验证k:")
    [k_f, k_l] = cross_validation(5, features, labels)
    bayes(k_f, k_l)
    knn_classifier(k_f, k_l)
