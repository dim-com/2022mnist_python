import numpy as np

from data.mnist_load import load_mnist


#  (Manhattan) distance版本
#  运行knn1.py可以输入测试集，训练集数目得到正确率

def knn_predict(tes, num):
    (train_imgs, train_labs), (_, _) = load_mnist(one_hot_label=True)
    # 加载数据集

    train_da = train_imgs[:num:]
    # 选择使用的训练集数量

    dis = np.argmin(np.sum(np.abs(tes - train_da), axis=1))
    # 计算L1距离
    rate = (num*2 - dis) / (num*2)

    predi = np.argmax(train_labs[dis])
    # 取得距离最近图片的标签

    return predi, rate


if __name__ == '__main__':

    (train_images, train_labels), (test_images, test_labels) = load_mnist(one_hot_label=True)

    test = int(input("test:"))
    ac = 0
    train = int(input("train:"))

    for i in range(test):
        test_data = test_images[i]
        train_data = train_images[:train, :]
        # 选择使用的训练集数量

        distance = np.argmin(np.sum(np.abs(test_data - train_data), axis=1))
        # 计算L1距离

        predict = np.argmax(train_labels[distance])
        # 取得距离最近图片的标签
        real = np.argmax(test_labels[i])
        # 取得真实标签

        if predict == real:
            # 判断真实和预测
            ac += 1
            # 准确数增加
        print(predict == real)
    print("训练集数量:", train)
    print("测试集数量:", test)
    print("准确率:", ac / test)
