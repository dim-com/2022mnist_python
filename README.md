pythonProject1为源代码，report为报告

1.运行源代码所需要执行的命令

1）程序界面运行main.py即可，操作见gui中的按钮。

2）运行knn1.py，输入测试集，训练集数目得到正确率，建议输入（50，2000）附近，可以显示每一个预测真确与否。

3）运行knn3.py，操作同上，但是此时k=3

4）运行nn.py，可以生成四个对图像进行不同操作的model文件，并且可以得到每一次训练的loss和acc，

以及测试集的loss和acc，后续可以在main当中调用。

5）运行cnn.py,可以生成一个卷积神经网络，同上，可以得到每一次训练的loss和acc。

*6）运行mnist_transfer_cnn.py,可以先生成一个由前五个数字训练的卷积神经网络，再得到整个数字的训练结果。*

*7）运行my_cnn.py，加入了使用plt绘制训练结果，得到一个pkl文件*

*8）运行nn_soothing,使用标签平滑优化模型，将数据集规模减小，并且不使用卷积网络，否则效果不突出*

9）运行mnist_load.py文件可以下载数据集，之后通过load_mnist()以pickle加载出来。

2.环境依赖

pillow,pyqt5,pyqt5_tools,numpy,tensorflow,plt

1）解释器版本为3.8，但是3.10也能跑，主要原因是3.10打不开qt designer。

2）Tensorflow：2.80 CUDA：11.1 cuDNN：8.1.0 RTX3060

需要注意，tensorflow版本和CUDA，cuDNN版本要相互匹配，不然无法训练，与提取。

3）ps：如还有错误可以查看report——软件包部分，或者麻烦老师联系我，联系方式见报告第一页，谢谢。