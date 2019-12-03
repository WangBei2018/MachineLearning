import tensorflow as tf
from time import time
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 导入mnist数据集
(train_image, train_label), (test_image, test_label) = tf.keras.datasets.mnist.load_data()
print('训练集信息：')
print(train_image.shape, train_label.shape)
print('测试集信息：')
print(test_image.shape, test_label.shape)

# 将原先的0-9label转换为独热编码
train_label_onehot = tf.keras.utils.to_categorical(train_label)
# print(train_label_onehot)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))        #格式化数据为28*28
model.add(tf.keras.layers.Dense(100, activation="relu"))        #第一层隐含层神经元个数,Dense函数：全连接
model.add(tf.keras.layers.Dense(10, activation="softmax"))      #输出神经元个数10类

# compile方法确定模型训练结构
# loss：损失函数 categorical_crossentropy：交叉熵
# optimizer：优化器 损失函数优化器：Adam loss:损失函数 lr:学习率
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
startTime = time()

text_label_onehot = tf.keras.utils.to_categorical(test_label)
# batch_size：每次梯度下降时包含的样本数 epochs:训练轮数
# validation_data：验证集
history = model.fit(train_image, train_label_onehot, batch_size=5000, epochs=100, validation_data=(test_image, text_label_onehot))
duration = time() - startTime
print('duration:', duration)
plt.plot(history.epoch, history.history.get('accuracy'), label='训练集准确率acc')
plt.plot(history.epoch, history.history.get('val_accuracy'), label='测试集准确率val_acc')
plt.xlabel("训练次数")
plt.ylabel("准确率")
plt.title("神经网络训练模型准确率变化示意图")
plt.legend()
plt.show()