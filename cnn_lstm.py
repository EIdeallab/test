import UtilStock
import datapreprocess
import pymssql as mssql
from keras.models import Sequential
from keras.layers import Input, Flatten, Dense, LSTM, Conv1D, TimeDistributed, Dropout
from keras.layers import LeakyReLU
import numpy as np
import math
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#data params
train_ratio = 0.7

# Convolution
kernel_size = 3
filters = 64
pool_size = 2

feature_num = 4
data_size = 5
date_size = 5

if __name__ == "__main__":
    server, user, password, database = UtilStock.ParseConfig('config.ini')
    connect = mssql.connect(server=server, user=user, password=password, database=database, charset='UTF8')
    cur = connect.cursor()
    info = UtilStock.LoadStockInfo(cur)
    data, label = datapreprocess.getInfoLabelto2DArray(cur, info, data_size= data_size, date_size= date_size, scaler=True)
    #data = np.reshape(data, (data.shape[0], 1, data.shape[1]*data.shape[2] ))

    train_Data = data[ 0 : int(len(data) * train_ratio)]
    train_Label = label[0 : int(len(data) * train_ratio)]

    test_Data = data[int(len(data) * train_ratio) : len(data)]
    test_Label = label[int(len(data) * train_ratio) : len(data)]

    model = Sequential()
    '''
    model.add(Conv1D(filters=32,
                   kernel_size=2,
                   strides=1,
                   input_shape=(date_size,feature_num,1)))

    model.add(LeakyReLU(alpha=0.2))
    model.add(Flatten())
    '''
    #model.add(LSTM(4, activation='tanh', input_shape=(date_size,feature_num)))
    model.add(LSTM(50, return_sequences=True, input_shape=(date_size, feature_num)))
    model.add(LSTM(64, return_sequences=False))
    #model.add(Dropout(0, 2))
    #model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='adam', metrics=['acc'])

    print('Model Build...')
    model.summary()

    print('Train...')
    model.fit(train_Data, train_Label,
              epochs=100,
              batch_size=32, verbose=2,
              validation_data=(test_Data, test_Label))

    testPredict = model.predict(test_Data)
    testScore = math.sqrt(mean_squared_error(test_Label, testPredict))
    print('Train Score: %.2f RMSE' % testScore)

    fig = plt.figure(facecolor='white', figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.plot(test_Label, label='True')
    ax.plot(testPredict, label='Prediction')
    ax.legend()
    plt.show()
    '''
    score, acc = model.evaluate(test_Data, test_Label)
    print('Test score:', score)
    print('Test accuracy:', acc)
    '''