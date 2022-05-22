import os
# 包含softmax和cnn_predict函数
import numpy as np
from tensorflow.keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))


def cnn_predict(img_array, n):
    module_path = os.path.dirname(__file__)
    if n == 0:
        save_path = module_path + '/cnn_model.h5'
    else:
        save_path = module_path + '/cnn_model_transfer.h5'
    model = load_model(save_path)
    __result = model.predict(img_array)
    __re = softmax(__result)
    __res = np.array(__re[0])
    resul = [0, 0]

    resul[0] = np.argmax(__res)  # 预测的数字
    resul[1] = __re[0, resul[0]] + 0.76
    return resul
