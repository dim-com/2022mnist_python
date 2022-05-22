"""Transfer learning toy example.
迁移学习实例
1 - Train a simple convnet on the MNIST dataset the first 5 digits [0..4].
1 - 基于MINIST数据集，训练简单卷积网络，前5个数字[0..4].
2 - Freeze convolutional layers and fine-tune dense layers
   for the classification of digits [5..9].
2 - 为[5..9]数字分类，冻结卷积层并微调全连接层
Get to 99.8% test accuracy after 5 epochs
for the first five digits classifier
and 99.2% for the last five digits after transfer + fine-tuning.
5个周期后，前5个数字分类测试准确率99.8% ，同时通过迁移+微调，后5个数字测试准确率99.2%
"""

from __future__ import print_function
import tensorflow as tf

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend

batch_size = 128
num_classes = 5  # 分类类别个数
epochs = 5  # 迭代次数

# 输入图像维度
img_rows, img_cols = 28, 28  # 28*28的像素
# 使用的卷积过滤器数量
filters = 32
# 最大值池化的池化区域大小
pool_size = 2
# 卷积核大小
kernel_size = 3

if backend.image_data_format() == 'channels_first':
    input_shape = (1, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 1)


def train_model(mod, train, test, num_cla):
    x_tra = train[0].reshape((train[0].shape[0],) + input_shape)
    x_tes = test[0].reshape((test[0].shape[0],) + input_shape)
    x_tra = x_tra.astype('float32')
    x_tes = x_tes.astype('float32')
    x_tra /= 255
    x_tes /= 255
    print('x_train shape:', x_tra.shape)
    print(x_tra.shape[0], 'train samples')
    print(x_tes.shape[0], 'test samples')

    # 类别向量转为多分类矩阵
    y_tra = tf.keras.utils.to_categorical(train[1], num_cla)
    y_tes = tf.keras.utils.to_categorical(test[1], num_cla)

    mod.compile(loss='categorical_crossentropy',
                optimizer='adadelta',
                metrics=['accuracy'])

    mod.fit(x_tra, y_tra,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_tes, y_tes))

    score = mod.evaluate(x_tes, y_tes, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


# 筛选（数据顺序打乱）、划分训练集和测试集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 创建2个数据集，一个数字小于5，另一个数学大于等与5
x_train_lt5 = x_train[y_train < 5]
y_train_lt5 = y_train[y_train < 5]
x_test_lt5 = x_test[y_test < 5]
y_test_lt5 = y_test[y_test < 5]

x_train_gte5 = x_train[y_train >= 5]
y_train_gte5 = y_train[y_train >= 5] - 5
x_test_gte5 = x_test[y_test >= 5]
y_test_gte5 = y_test[y_test >= 5] - 5

# 定义2组层：特征（卷积）和分类（全连接）
# 特征 = Conv + relu + Conv + relu + pooling + dropout
feature_layers = [
    Conv2D(filters, kernel_size,
           padding='valid',
           input_shape=input_shape),
    Activation('relu'),
    Conv2D(filters, kernel_size),
    Activation('relu'),
    MaxPooling2D(pool_size=pool_size),
    Dropout(0.25),
    Flatten(),
]

# 分类 = 128全连接 + relu + dropout + 5全连接 + softmax
classification_layers = [
    Dense(128),
    Activation('relu'),
    Dropout(0.5),
    Dense(num_classes),
    Activation('softmax')
]

# 创建完整模型
model = Sequential(feature_layers + classification_layers)

# 为5数字分类[0..4]训练模型
train_model(model,
            (x_train_lt5, y_train_lt5),
            (x_test_lt5, y_test_lt5), num_classes)

# 冻结上层并重建模型
for x in feature_layers:
    x.trainable = False

# 迁移：训练下层为[5..9]分类任务
train_model(model,
            (x_train_gte5, y_train_gte5),
            (x_test_gte5, y_test_gte5), num_classes)
model.save('cnn_model_transfer.h5')
del model
