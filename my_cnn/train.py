import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from data.mnist_load import load_mnist
from my_cnn import MyCnn
from my_fun.classes import Trainer


sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
# 读入全部数据
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)
network = MyCnn()
trainer = Trainer(network, x_train, t_train, x_test, t_test)
trainer.train()

# 保存参数
network.save_params("my_model.pkl")
print("--Saved--")

# 绘制图形
markers = {'train': 'o', 'test': 's'}
x = np.arange(20)
plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
