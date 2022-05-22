# keras.datasets 给予 mnist 数据
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.python.keras.utils import np_utils


def label_smoothing(numOfHiddenNodes, Parameter):
    nb_classes = 10
    # load train and test data
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = loadData()

    # convert Y_train to a hot vector
    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    Y_test = np_utils.to_categorical(Y_test, nb_classes)
    Y_valid = np_utils.to_categorical(Y_valid, nb_classes)

    optim = Adam(lr=0.001)

    # smooth labels
    prob = Parameter * 1.0 / (nb_classes - 1)
    for yIndex, dum1 in enumerate(Y_train):
        for yprob, dum2 in enumerate(Y_train[yIndex, :]):
            if Y_train[yIndex, yprob] == 1:
                Y_train[yIndex, yprob] = 1 - Parameter
            else:
                Y_train[yIndex, yprob] = prob

    model = Sequential()
    model.add(Dense(numOfHiddenNodes, input_shape=(28 * 28,)))
    model.add(Activation('relu'))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])

    model.fit(X_train, Y_train,
              batch_size=100, verbose=1,
              validation_data=(X_valid, Y_valid))

    score = model.evaluate(X_test, Y_test,
                           verbose=1)
    model.save(f'my_model4.h5')
    # creates a HDF5 file 'my_model.h5'
    # 根据官方文档，不建议使用其他文件格式储存
    del model
    return score[0], score[1]


def loadData():
    numOfValidation = 10000
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    X_train = X_train.reshape(len(X_train), 28 * 28)
    X_test = X_test.reshape(len(X_test), 28 * 28)
    # normalize data（归一化）
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    X_valid = X_train[0: numOfValidation, :]
    Y_valid = Y_train[0: numOfValidation]

    return X_train[numOfValidation:, :], Y_train[numOfValidation:, ], X_valid, Y_valid, X_test, Y_test


original_acc_vector = []
smoothing_acc_vector = []
for loop in range(1, 10):
    original_acc, smoothing_acc = label_smoothing(numOfHiddenNodes=100, Parameter=0.01)
    original_acc_vector.append(original_acc)
    smoothing_acc_vector.append(smoothing_acc)

loss = sum(original_acc_vector) * 100.0 / len(original_acc_vector)
smoothingMean = sum(smoothing_acc_vector) * 100.0 / len(smoothing_acc_vector)
print('acc:')
print('Loss:', loss)
print('label smoothing:', smoothingMean)
