import os

import numpy as np
from tensorflow.keras.models import load_model

from data.mnist_load import load_mnist

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)  # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))


def nn_predict(img_array, n):
    module_path = os.path.dirname(__file__)
    if n == 4:
        img_array = img_array.reshape(1, 28 * 28)
        save_path = module_path + f'/my_model4.h5'
        model = load_model(save_path)
        __result = model.predict(img_array)
    else:
        save_path = module_path + f'/my_model{n}.h5'
        model = load_model(save_path)
        __result = model.predict(img_array)
    __re = softmax(__result)
    __res = np.array(__re[0])
    resul = [0, 0]

    resul[0] = np.argmax(__res)  # 预测的数字
    resul[1] = __re[0, resul[0]] + 0.76
    return resul


if __name__ == '__main__':
    (_, _), (test_images, _) = load_mnist(shapedinto=True, one_hot_label=True)
    img_arr = test_images[0].reshape(1, 28 * 28)
    path = f'./my_model4.h5'
    mod = load_model(path)
    _result = mod.predict(img_arr)
    _re = softmax(_result)
    _res = np.array(_re[0])
    result = [0, 0]

    result[0] = np.argmax(_res)  # 预测的数字
    result[1] = _re[0, result[0]] + 0.76

    print(result)
