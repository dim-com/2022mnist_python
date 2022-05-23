import os
import sys
import warnings

import numpy as np
from PIL import Image, ImageQt, ImageFilter
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QPixmap, QColor
# QImage,ImageQt也许可以省略一个
from PyQt5.QtWidgets import QMainWindow, QDesktopWidget, QApplication, QFileDialog
from PyQt5.QtWidgets import QMessageBox

import cnn.fun
import knn.knn1
import knn.knn3
import nn.fun
from data.mnist_load import load_mnist
# 有一个警告需要忽略
# 下面import文件里面的
from my_cnn.my_cnn import MyCnn
from qt.layout import Ui_MainWindow
from qt.paintboard import PaintBoard

(_, _), (x_test, _) = load_mnist(flatten=False)
# 读取MNIST数据集,用来抽取展示
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 有一个警告需要忽略
# my_cnn初始化
module_path = os.path.dirname(__file__)
network = MyCnn()
path = module_path + "/my_cnn/my_model.pkl"
network.load_params(path)


class MainWindow(QMainWindow, Ui_MainWindow):
    image_number = 1

    def __init__(self):
        super(MainWindow, self).__init__()
        # 参数
        self.mode = 1
        self.mo = 0
        self.mod = 10000
        self.mod1 = 0

        self.result = [0, 0]
        # UI
        self.setupUi(self)
        self.center()
        self.pbtGetfile.setEnabled(False)
        # 画板
        self.paintBoard = PaintBoard(self, Size=QSize(224, 224), Fill=QColor(0, 0, 0, 0))
        self.paintBoard.setPenColor(QColor(0, 0, 0, 0))
        self.dArea_Layout.addWidget(self.paintBoard)

        self.clearDataArea()

    def center(self):
        # 获得窗口
        framePos = self.frameGeometry()
        # 获得屏幕中心点
        scPos = QDesktopWidget().availableGeometry().center()
        # 显示到屏幕中心
        framePos.moveCenter(scPos)
        self.move(framePos.topLeft())

    # 窗口关闭
    def closeEvent(self, event):
        reply = QMessageBox.question(self, '退出',
                                     "确定要退出吗?", QMessageBox.Yes |
                                     QMessageBox.No, QMessageBox.Yes)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()
            # 清除数据待输入区

    def clearDataArea(self):
        self.paintBoard.Clear()
        self.lbDataArea.clear()
        self.lbResult.clear()
        self.lbCofidence.clear()
        self.result = [0, 0]

    # 模式下拉列表回调
    def cbBox_Mode_Callback(self, text):
        if text == '1：测试集抽取':
            self.mode = 1
            self.clearDataArea()
            self.pbtGetMnist.setEnabled(True)
            self.pbtGetfile.setEnabled(False)
            self.pbtFilter1.setEnabled(True)
            self.pbtFilter2.setEnabled(True)
            self.pbtFilter3.setEnabled(True)
            self.paintBoard.setBoardFill(QColor(0, 0, 0, 0))
            self.paintBoard.setPenColor(QColor(0, 0, 0, 0))

        elif text == '2：鼠标手写输入':
            self.mode = 2
            self.clearDataArea()
            self.pbtGetMnist.setEnabled(False)
            self.pbtGetfile.setEnabled(False)
            self.pbtFilter1.setEnabled(False)
            self.pbtFilter2.setEnabled(False)
            self.pbtFilter3.setEnabled(False)
            self.paintBoard.setBoardFill(QColor(0, 0, 0, 255))
            self.paintBoard.setPenColor(QColor(255, 255, 255, 255))

        elif text == '3：打开本地文件':
            self.mode = 3
            self.clearDataArea()
            self.pbtGetMnist.setEnabled(False)
            self.pbtGetfile.setEnabled(True)
            self.pbtFilter1.setEnabled(True)
            self.pbtFilter2.setEnabled(True)
            self.pbtFilter3.setEnabled(True)
            self.paintBoard.setBoardFill(QColor(0, 0, 0, 0))
            self.paintBoard.setPenColor(QColor(0, 0, 0, 0))

    def cbBox_Mode_Callback_2(self, text):
        if text == '0：原始数据':
            self.mo = 0
        elif text == '1：均值滤波':
            self.mo = 1
        elif text == '2：边缘检测':
            self.mo = 2
        elif text == '3：轮廓显示':
            self.mo = 3
        elif text == '4：标签平滑':
            self.mo = 4

    def cbBox_Mode_Callback_3(self, text):
        if text == 'size:100':
            self.mod = 100
        elif text == 'size:1000':
            self.mod = 1000
        elif text == 'size:10000':
            self.mod = 10000
        elif text == 'size:50000':
            self.mod = 50000

    def cbBox_Mode_Callback_4(self, text):
        if text == 'Keras模型':
            self.mod = 0
        elif text == 'Keras_transfer':
            self.mod = 1
        elif text == 'my_cnn':
            self.mod = 2

    # 数据清除
    def pbtClear_Callback(self):
        self.clearDataArea()

    # KNN识别
    def pbtPredict_Callback(self):
        __img = []
        # 获取qimage图像
        if self.mode == 2:
            __img = self.paintBoard.getContentAsQImage()
        else:
            __img = self.lbDataArea.pixmap()
            # label内无图像返回None
            if __img is None:
                # 无图像为纯黑
                __img = ImageQt.ImageQt(Image.fromarray(np.uint8(np.zeros([224, 224]))))
            else:
                __img = __img.toImage()

        # 转换成pil image类型处理
        pil_img = ImageQt.fromqimage(__img)
        pil_img = pil_img.resize((28, 28), Image.ANTIALIAS)
        img_array = np.array(pil_img.convert('L'))
        img_array = img_array.reshape(1, -1) / 255

        __result = knn.knn1.knn_predict(img_array, self.mod)

        self.result[0] = __result[0]
        # 预测的数字
        self.result[1] = __result[1]
        # 置信度

        self.lbResult.setText("%d" % (self.result[0]))
        self.lbCofidence.setText("%.4f" % (self.result[1]))

    def pbtPredict_1_Callback(self):
        __img = []
        # 获取qimage图像
        if self.mode == 2:
            __img = self.paintBoard.getContentAsQImage()
        else:
            __img = self.lbDataArea.pixmap()
            # label内无图像返回None
            if __img is None:
                # 无图像为纯黑
                __img = ImageQt.ImageQt(Image.fromarray(np.uint8(np.zeros([224, 224]))))
            else:
                __img = __img.toImage()

        # 转换成pil image类型处理
        pil_img = ImageQt.fromqimage(__img)
        pil_img = pil_img.resize((28, 28), Image.ANTIALIAS)
        img_array = np.array(pil_img.convert('L'))
        img_array = img_array.reshape(1, -1) / 255

        __result = knn.knn3.knn_predict(img_array, self.mod)

        self.result[0] = __result[0]
        # 预测的数字
        self.result[1] = __result[1]
        # 置信度

        self.lbResult.setText("%d" % (self.result[0]))
        self.lbCofidence.setText("%.4f" % (self.result[1]))

    # nn
    def pbtPredict_2_Callback(self):
        __img = []
        # 获取qimage图像
        if self.mode == 2:
            __img = self.paintBoard.getContentAsQImage()
        else:
            __img = self.lbDataArea.pixmap()
            # label内无图像返回None
            if __img is None:
                # 无图像为纯黑
                __img = ImageQt.ImageQt(Image.fromarray(np.uint8(np.zeros([224, 224]))))
            else:
                __img = __img.toImage()

        # 转换成pil image类型处理
        pil_img = ImageQt.fromqimage(__img)
        pil_img = pil_img.resize((28, 28), Image.ANTIALIAS)
        img_array = np.array(pil_img.convert('L')).reshape(1, 28, 28)
        img_array = img_array / 255

        __result = nn.fun.nn_predict(img_array, self.mo)

        self.result[0] = __result[0]
        # 预测的数字
        self.result[1] = __result[1]
        # 置信度

        self.lbResult.setText("%d" % (self.result[0]))
        self.lbCofidence.setText("%.4f" % (self.result[1]))

    def pbtPredict_3_Callback(self):
        __img = []
        # 获取qimage图像
        if self.mode == 2:
            __img = self.paintBoard.getContentAsQImage()
        else:
            __img = self.lbDataArea.pixmap()
            # label内无图像返回None
            if __img is None:
                # 无图像为纯黑
                __img = ImageQt.ImageQt(Image.fromarray(np.uint8(np.zeros([224, 224]))))
            else:
                __img = __img.toImage()

        # 转换成pil image类型处理
        pil_img = ImageQt.fromqimage(__img)
        pil_img = pil_img.resize((28, 28), Image.ANTIALIAS)

        if self.mod1 < 2:
            img_array = np.array(pil_img.convert('L')).reshape(1, 28, 28, 1)
            img_array = img_array / 255
            __result = cnn.fun.cnn_predict(img_array, self.mod1)

        else:
            img_array = np.array(pil_img.convert('L')).reshape(1, 1, 28, 28) / 255.0
            # img_array = np.where(img_array>0.5, 1, 0)

            # reshape成网络输入类型
            __result = network.predict(img_array)  # shape:[1, 10]
            # 将预测结果使用softmax输出
            __result = cnn.fun.softmax(__result)
        self.result[0] = __result[0]
        # 预测的数字
        self.result[1] = __result[1]
        # 置信度

        self.lbResult.setText("%d" % (self.result[0]))
        self.lbCofidence.setText("%.4f" % (self.result[1]))

    def pbtSave_Callback(self):
        __img = []
        # 获取qimage图像
        if self.mode == 2:
            __img = self.paintBoard.getContentAsQImage()
        else:
            __img = self.lbDataArea.pixmap()  # label内若无图像返回None
            if __img is None:  # 无图像为纯黑
                __img = ImageQt.ImageQt(Image.fromarray(np.uint8(np.zeros([224, 224]))))
            else:
                __img = __img.toImage()

        # 转换成pil image
        im = ImageQt.fromqimage(__img)
        im = im.convert('RGB')
        im.save(f'./save{self.image_number}.jpg')

    def pbtGetfile_Callback(self):
        self.clearDataArea()

        filename, _ = QFileDialog.getOpenFileName(self, '打开文件', "", "All Files(*)")
        pil_img = Image.open(filename)
        pil_img = pil_img.resize((28, 28))
        img_array = pil_img.convert('L')
        img = img_array.resize((224, 224))
        # 放大
        qimage = ImageQt.ImageQt(img)

        # 将qimage类型图像显示在label
        pix = QPixmap.fromImage(qimage)
        self.lbDataArea.setPixmap(pix)

    # 随机抽取，只是显示用的
    def pbtGetMnist_Callback(self):
        self.clearDataArea()

        # 随机抽取一张测试集图片，放大后显示
        img = x_test[np.random.randint(0, 9999)]
        # [1,28,28]
        img = img.reshape(28, 28)
        # [28,28]

        img = img * 0xff  # 恢复灰度值大小
        pil_img = Image.fromarray(np.uint8(img))
        pil_img = pil_img.resize((224, 224))  # 放大

        # pil转换成qimage
        qimage = ImageQt.ImageQt(pil_img)

        # 将qimage显示在label
        pix = QPixmap.fromImage(qimage)
        self.lbDataArea.setPixmap(pix)

    def pbtFilter1_Callback(self):  # 均值滤波
        __img = []
        # 获取qimage图像
        if self.mode == 2:
            __img = self.paintBoard.getContentAsQImage()
        else:
            __img = self.lbDataArea.pixmap()  # label内若无图像返回None
            if __img is None:  # 无图像为纯黑
                __img = ImageQt.ImageQt(Image.fromarray(np.uint8(np.zeros([224, 224]))))
            else:
                __img = __img.toImage()

        # 转换成pil image
        im = ImageQt.fromqimage(__img)
        im = im.filter(ImageFilter.BLUR)
        pil_img = im.resize((28, 28))
        img_array = pil_img.convert('L')
        img = img_array.resize((224, 224))
        # 放大
        qimage = ImageQt.ImageQt(img)

        # 将qimage类型图像显示在label
        pix = QPixmap.fromImage(qimage)
        self.lbDataArea.setPixmap(pix)

    def pbtFilter2_Callback(self):  # 边缘检测
        __img = []
        # 获取qimage图像
        if self.mode == 2:
            __img = self.paintBoard.getContentAsQImage()
        else:
            __img = self.lbDataArea.pixmap()  # label内若无图像返回None
            if __img is None:  # 无图像为纯黑
                __img = ImageQt.ImageQt(Image.fromarray(np.uint8(np.zeros([224, 224]))))
            else:
                __img = __img.toImage()

        # 转换成pil image
        im = ImageQt.fromqimage(__img)
        im = im.filter(ImageFilter.EDGE_ENHANCE_MORE)
        pil_img = im.resize((28, 28))
        img_array = pil_img.convert('L')
        img = img_array.resize((224, 224))
        # 放大
        qimage = ImageQt.ImageQt(img)

        # 将qimage类型图像显示在label
        pix = QPixmap.fromImage(qimage)
        self.lbDataArea.setPixmap(pix)

    def pbtFilter3_Callback(self):  # 轮廓显示
        __img = []
        # 获取qimage图像
        if self.mode == 2:
            __img = self.paintBoard.getContentAsQImage()
        else:
            __img = self.lbDataArea.pixmap()  # label内若无图像返回None
            if __img is None:  # 无图像为纯黑
                __img = ImageQt.ImageQt(Image.fromarray(np.uint8(np.zeros([224, 224]))))
            else:
                __img = __img.toImage()

        # 转换成pil image
        im = ImageQt.fromqimage(__img)
        im = im.filter(ImageFilter.CONTOUR)
        pil_img = im.resize((28, 28))
        img_array = pil_img.convert('L')
        img = img_array.resize((224, 224))
        # 放大
        qimage = ImageQt.ImageQt(img)

        # 将qimage类型图像显示在label
        pix = QPixmap.fromImage(qimage)
        self.lbDataArea.setPixmap(pix)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    Gui = MainWindow()
    Gui.show()
    sys.exit(app.exec_())
