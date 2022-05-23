# 这个基本是依靠tensorflow的,用于训练
import os
import warnings
import numpy as np
import tensorflow as tf
from PIL import Image, ImageFilter
from data.mnist_load import load_mnist

warnings.filterwarnings("ignore")
# 有一个警告需要忽略

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 又是一个要忽略的报错
(train_images, train_labels), (test_images, test_labels) = load_mnist(shapedinto=True, one_hot_label=True)
times = 0


def train(train_image, train_label, test_image, test_label):
    global times
    model = tf.keras.Sequential()
    # 顺序结构的模型
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(64, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    # 设置优化器，损失函数和记录准确率
    history = model.fit(train_image, train_label, epochs=10, validation_data=(test_image, test_label))
    # 载入训练集、验证级、设置训练轮次
    model.evaluate(test_image, test_label)
    # 使用测试集进行评估
    model.save(f'my_model{times}.h5')
    # creates a HDF5 file 'my_model.h5'
    # 根据官方文档，不建议使用其他文件格式储存
    del model

    times = times + 1


train(train_images, train_labels, test_images, test_labels)


def transform(way):
    for x in train_images:
        x = x * 0xff  # 恢复灰度值大小
        x = Image.fromarray(np.uint8(x))
        x = x.filter(way)
        x = x.resize((28, 28), Image.ANTIALIAS)
        x = np.array(x.convert('L'))
        x = x / 255
    for x in test_images:
        x = x * 0xff  # 恢复灰度值大小
        x = Image.fromarray(np.uint8(x))
        x = x.filter(way)
        x = x.resize((28, 28), Image.ANTIALIAS)
        x = np.array(x.convert('L'))
        x = x / 255


transform(ImageFilter.BLUR)
train(train_images, train_labels, test_images, test_labels)
transform(ImageFilter.EDGE_ENHANCE_MORE)
train(train_images, train_labels, test_images, test_labels)
transform(ImageFilter.CONTOUR)
train(train_images, train_labels, test_images, test_labels)
