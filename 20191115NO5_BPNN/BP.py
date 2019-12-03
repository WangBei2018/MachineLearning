import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


def read_excel():
    rawdata = pd.read_csv('../data/2-mnist.csv')
    rawdata = rawdata.values
    labels = rawdata[:, 0]
    features = rawdata[:, 1:]
    data = rawdata[3, 1:].reshape(28, 28)
    plt.imshow(data, cmap='gray')
    plt.show()

    return labels, features


def show():
    return


def bp(labels, features):

    return


if __name__ == '__main__':
    [labels, features] = read_excel()
    bp(labels, features)