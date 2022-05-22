# cnn，用keras的cnn，第三次作业可以尝试自己写一个
import os
import warnings

import tensorflow as tf

from data.mnist_load import load_mnist

warnings.filterwarnings("ignore")
# 有一个警告需要忽略

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 又是一个要忽略的报错
(train_images, train_labels), (test_images, test_labels) = load_mnist(shapedcnn=True, one_hot_label=True)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), input_shape=(28, 28, 1), padding='same',
                                 activation="sigmoid"))
model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), activation="sigmoid"))
model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(filters=120, kernel_size=(5, 5), activation="sigmoid"))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(84, activation='sigmoid'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
model.evaluate(test_images, test_labels)
model.save('cnn_model.h5')
del model
