import numpy as np


def load_npy(group, dir_path="dataset/"):
    group_info = {"1": (17731, 100),
                  "2": (46219, 259),
                  "3": (21820, 100),
                  "4": (54028, 248)}
    train_num, test_num = group_info[str(group)]
    print("train_num:{} test_num:{}".format(train_num, test_num))

    xtrain_norm = np.load(dir_path + "{}/{}xtrain_norm{}_28_3_17.npy".format(group, group, train_num))
    ytrain = np.load(dir_path + "{}/{}ytrain{}.npy".format(group, group, train_num))
    xtest_norm = np.load(dir_path + "{}/{}xtest_norm{}_28_3_17.npy".format(group, group, test_num))
    ytest = np.load(dir_path + "{}/{}ytest{}.npy".format(group, group, test_num))

    return xtrain_norm, ytrain, xtest_norm, ytest


GROUP = 1
PATH = "dataset/"
xtrain, ytrain, xtest, ytest = load_npy(GROUP, PATH)
# xtrain=xtrain[:, :, :, 3:]
# xtest=xtest[:, :, :, 3:]

from tensorflow.keras import Input, Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import TimeDistributed, BatchNormalization, Activation, Concatenate, LSTM, Dense, Conv1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)



class MCNNLSTM(Model):
    def __init__(self):
        super(MCNNLSTM, self).__init__()
        self.cnn1 = Sequential(
            [TimeDistributed(Conv1D(filters=7, kernel_size=3, padding='same'), input_shape=(28, 3, 1)),
             TimeDistributed(BatchNormalization()), TimeDistributed(Activation('relu'))])
        self.cnn2 = Sequential([TimeDistributed(Conv1D(filters=7, kernel_size=3, padding='same')),
                                TimeDistributed(BatchNormalization()), TimeDistributed(Activation('relu'))])
        self.flatten = TimeDistributed(Flatten())
        self.lstm1 = LSTM(units=400, return_sequences=True)
        self.lstm2 = LSTM(units=200, return_sequences=False)
        self.out = Dense(1, activation='linear', name='main_output')

    def call(self, raw):
        multi_head = []
        for i in range(raw.shape[-1]):
            x = raw[:, :, :, i]
            x = tf.expand_dims(x, axis=-1)
            x = self.cnn1(x)
            x = self.cnn2(x)
            x = self.flatten(x)
            multi_head.append(x)
        combined = Concatenate(axis=-1)([x for x in multi_head])  # (None, n_window=28, 21*17)

        x = self.lstm1(combined)
        x = self.lstm2(x)
        x = self.out(x)
        y = tf.squeeze(input=x)

        return y


def scheduler(epoch, lr):
    if epoch == 10:
        return lr * 0.1
    elif epoch == 15:
        return lr * 0.1
    elif epoch == 20:
        return lr * tf.math.exp(-0.1)
    else:
        return lr


model = MCNNLSTM()
optimizers = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9, centered=True)
# optimizers=tf.keras.optimizers.SGD(lr=0.001)
lr_scheduler = LearningRateScheduler(scheduler)
k_rmse=tf.keras.metrics.RootMeanSquaredError()
model.compile(optimizer=optimizers, loss='mse', metrics=[k_rmse,'mse'])
model.fit(xtrain, ytrain, epochs=30, batch_size=400,
          callbacks=[lr_scheduler, EarlyStopping(monitor='mse', min_delta=0,
                                                     patience=20, verbose=1, mode='min')])

# ypred = model(xtrain[:64,:,:,:])
# print(ypred.shape)


from sklearn.metrics import mean_squared_error

MAX_RUL = 130

def myScore(y_ture, y_pred):
    score = 0
    for i in range(len(y_pred)):
        if y_ture[i] <= y_pred[i]:
            score = score + np.exp(-(y_ture[i] - y_pred[i]) / 10.0) - 1
        else:
            score = score + np.exp((y_ture[i] - y_pred[i]) / 13.0) - 1
    return score


yPreds = model.predict(xtest)
yPreds = yPreds * MAX_RUL
yPreds = yPreds.ravel()
test_rmse = np.sqrt(mean_squared_error(ytest, yPreds))
test_score = myScore(ytest, yPreds)
print('lastScore:', test_score, 'lastRMSE', test_rmse)

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

plt.plot(yPreds, c='green')
plt.plot(ytest, c='orange', linestyle='--')
plt.title("FD00{} RMSE:  {:.2f}  Score: {:.1f}".format(GROUP, test_rmse, test_score))
plt.xlabel('ID')
plt.ylabel('RUL')
plt.legend(["pred", "real"], loc='upper left')
plt.savefig("FD{}_score.png".format(GROUP))
plt.show()
