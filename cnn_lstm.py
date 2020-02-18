import UtilStock
import datapreprocess
import pymssql as mssql
import keras
from keras.models import Sequential
from keras.layers import Input, Flatten, Dense, LSTM, Conv1D, Conv2D, TimeDistributed, Dropout, ConvLSTM2D, BatchNormalization, Conv3D
from keras.layers import LeakyReLU
import numpy as np
import math
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint

from keras.callbacks import EarlyStopping
from keras.models import load_model
import os

num_classes = 5

#data params
train_ratio = 0.7
feature_num = 19
sample_size = 100
date_size = 5

CATEGORICAL = True

#model save path
MODEL_SAVE_FOLDER_PATH = './model/'
MODEL_NAME = 'LSTM'

#Use get ~to2DArray
def Load_Lstm_Model():

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(date_size, feature_num)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0, 2))
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.2))
    if CATEGORICAL:
        model.add(Dense(num_classes, activation='softmax'))
    else:
        model.add(Dense(1, activation='linear'))


    return model


#Use get ~to3DArray
def Load_Conv1D_Lstm_Model():

    model = Sequential()
    model.add(Conv1D(filters=32,
               kernel_size=3,
               strides=1,
               input_shape=(date_size, feature_num)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv1D(filters=64,kernel_size=3, strides=1))
    model.add(LeakyReLU(alpha=0.2))
    model.add(LSTM(64, return_sequences=True, activation='tanh'))
    model.add(Dropout(0, 2))
    model.add(LSTM(32, return_sequences=False, activation='tanh'))
    model.add(Dropout(0, 2))
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.2))
    if CATEGORICAL:
        model.add(Dense(num_classes, activation='softmax'))
    else:
        model.add(Dense(1, activation='linear'))

    return model

#Use get ~to3DArray
def Load_Deep_Conv1D_Lstm_Model():
    print('You must Use Large Dataset!!!!!!!!!')
    model = Sequential()
    model.add(Conv1D(filters=32,
               kernel_size=1,
               strides=1,
               input_shape=(date_size, feature_num)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv1D(filters=64,kernel_size=3, strides=1))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv1D(filters=128,kernel_size=1, strides=1))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv1D(filters=256,kernel_size=3, strides=1))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv1D(filters=512,kernel_size=1, strides=1))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(LSTM(128, return_sequences=True, activation='tanh'))
    model.add(Dropout(0, 2))
    model.add(LSTM(64, return_sequences=False, activation='tanh'))
    model.add(Dropout(0, 2))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(64, activation='linear'))
    if CATEGORICAL:
        model.add(Dense(num_classes, activation='softmax'))
    else:
        model.add(Dense(1, activation='linear'))

    return model

#Use get~to4DArray
def Load_Conv2DTD_Lstm_Model():

    model = Sequential()
    model.add(TimeDistributed(Conv2D(32, (2, 2), padding='same'), input_shape=(None, 3, 6, 1)))
    model.add(TimeDistributed(LeakyReLU(alpha=0.2)))
    model.add(TimeDistributed(Conv2D(64, (2, 2), padding='same')))
    model.add(TimeDistributed(LeakyReLU(alpha=0.2)))
    model.add(TimeDistributed(Conv2D(128, (2, 2))))
    model.add(TimeDistributed(LeakyReLU(alpha=0.2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(32, return_sequences=False, activation='tanh'))
    model.add(Dropout(0, 2))
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.2))

    if CATEGORICAL:
        model.add(Dense(num_classes, activation='softmax'))
    else:
        model.add(Dense(1, activation='linear'))
    return  model

#Use get~to4DArray
#reshape이 들어가기 때문에 옮기지 않음
def train_Conv2DLstm(data, label):

    train_Data = data[0: int(len(data) * train_ratio)]
    train_Label = label[0: int(len(data) * train_ratio)]
    train_Data = np.reshape(train_Data, (train_Data.shape[0],  train_Data.shape[1], 1,train_Data.shape[2],train_Data.shape[3]) )
    #train_Label = np.reshape(train_Label, (train_Label.shape[0],train_Label.shape[1], 1))

    test_Data = data[int(len(data) * train_ratio): len(data)]
    test_Label = label[int(len(data) * train_ratio): len(data)]
    test_Data = np.reshape(test_Data, (test_Data.shape[0], test_Data.shape[1], 1,test_Data.shape[2], test_Data.shape[3]))
    #test_Label = np.reshape(test_Label, (test_Label.shape[0], test_Label.shape[1], 1))

    model = Sequential()
    model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                       input_shape=(date_size, 1, 3, 6), #day/ - / feature/ feature
                       padding='same', return_sequences=True))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                       padding='same', return_sequences=True))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                       padding='same', return_sequences=True))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                       padding='same', return_sequences=True))
    model.add(BatchNormalization())
    model.add(Conv3D(filters=1, kernel_size=(3, 3, 1),padding='same', data_format='channels_last'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mse', optimizer='rmsprop', metrics=['acc'])

    print('Model Build...')
    model.summary()

    print('Train...')
    model.fit(train_Data, train_Label,
              epochs=5,
              batch_size=32, verbose=2,
              shuffle=False,
              validation_data=(test_Data, test_Label))

    testPredict = model.predict(test_Data)
    #testScore = math.sqrt(mean_squared_error(test_Label, testPredict))
    #print('Train Score: %.2f RMSE' % testScore)

    fig = plt.figure(facecolor='white', figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.plot(test_Label, label='True')
    ax.plot(testPredict, label='Prediction')
    ax.legend()
    plt.show()

def train_model(model,data,label):

    train_Data = data[0: int(len(data) * train_ratio)]
    train_Label = label[0: int(len(data) * train_ratio)]

    test_Data = data[int(len(data) * train_ratio): len(data)]
    test_Label = label[int(len(data) * train_ratio): len(data)]

    if CATEGORICAL:
        train_Label = keras.utils.to_categorical(train_Label, num_classes)
        test_Label = keras.utils.to_categorical(test_Label, num_classes)

    if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
        os.mkdir(MODEL_SAVE_FOLDER_PATH)

    model_path = MODEL_SAVE_FOLDER_PATH + MODEL_NAME + '{epoch:02d}-{val_loss:.4f}.h5'

    cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss',
                                    verbose=1, save_best_only=True)

    #early stopping이 필요할 때만 사용할 것
    #cb_early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    print('Train...')
    model.fit(train_Data, train_Label,
              epochs=100,
              batch_size=32, verbose=2,
              shuffle=True,
              validation_data=(test_Data, test_Label),
              callbacks=[cb_checkpoint])

    if CATEGORICAL:
        score = model.evaluate(test_Data, test_Label, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    else:
        testPredict = model.predict(test_Data)
        testScore = math.sqrt(mean_squared_error(test_Label, testPredict))
        print('Train Score: %.2f RMSE' % testScore)

        fig = plt.figure(facecolor='white', figsize=(10, 5))
        ax = fig.add_subplot(111)
        ax.plot(test_Label, label='True')
        ax.plot(testPredict, label='Prediction')
        ax.legend()
        plt.show()



if __name__ == "__main__":
    server, user, password, database = UtilStock.ParseConfig('config.ini')
    connect = mssql.connect(server=server, user=user, password=password, database=database, charset='UTF8')
    cur = connect.cursor()
    info = UtilStock.LoadFinanceStockInfo(cur)
    data, label = datapreprocess.getFinanceInfoLabelto3DArray(cur, info, data_size= sample_size, date_size= date_size, scaler=True, unit='WEEK', bLevel= True)
    # model =load_model('LSTM01-0.0030.hd5')
    model = Load_Lstm_Model()

    if CATEGORICAL:
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
    else:
        model.compile(loss='mse', optimizer='rmsprop', metrics=['acc'])
    print('Model Build...')
    model.summary()

    train_model(model, data, label)
