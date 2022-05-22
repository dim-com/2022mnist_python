import sys
import os


from data.mnist_load import load_mnist
from deep_convnet import DeepConvNet
from myfun.trainer import Trainer

sys.path.append(os.pardir)  # 为了导入父目录而进行的设定
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

network = DeepConvNet()  
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=20, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

# 保存参数
network.save_params("deep_convnet_params.pkl")
print("Saved Network Parameters!")
