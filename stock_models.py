import UtilStock
import datapreprocess
import pymssql as mssql
from keras.models import Sequential
from keras.layers import Flatten, Dense, LSTM, Conv1D, Conv2D, TimeDistributed, Dropout, ConvLSTM2D, BatchNormalization, Conv3D, GlobalAveragePooling2D, Concatenate,  Input
from keras.layers import LeakyReLU
import numpy as np
import matplotlib.pyplot as plt
import keras.applications.densenet as DenseNet

#optional
from keras.callbacks import EarlyStopping
from keras.models import load_model


#################################################
#### Do it here

#data params
train_ratio = 0.9
feature_num = 19
sample_size = 1000
date_size = 5
UNIT = 'WEEK'
SCALER = True
CATEGORICAL = True

#train params
train_epochs = 500

#model save path
MODEL_SAVE_FOLDER_PATH = './model/'
MODEL_NAME = 'CONV2DTD_LSTM'    # choose LSTM /  CONV1D_LSTM / DEEP_CONV1D_LSTM /  CONV2DTD_LSTM / DENSE121_LSTM / CONV2DTD_LSTM_IMG

#############################################



# if UNIT == 'DAY':
#     num_classes = 12
# elif UNIT == 'WEEK':
#     num_classes = 14


if UNIT == 'DAY':
    num_classes = 2
elif UNIT == 'WEEK':
    num_classes = 2


if(MODEL_NAME == 'CONV2DTD_LSTM_IMG'):
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    set_session(tf.Session(config=config))

class STOCK_MODEL:

    feature_num = 0
    sample_size = 0
    date_size = 0
    UNIT = ''
    SCALER = True
    CATEGORICAL = True

    num_classes =0

    def __init__(self, _feature_num, _sample_size, _date_size, _UNIT, _SCALER, _CATEGORICAL, _num_classes,  _MODEL_NAME):

        self.feature_num = _feature_num
        self.sample_size = _sample_size
        self.date_size = _date_size
        self.UNIT = _UNIT
        self.SCALER = _SCALER
        self.CATEGORICAL = _CATEGORICAL
        self.num_classes = _num_classes
        self.MODEL_NAME = _MODEL_NAME

        # Use get ~to2DArray
    def Load_Lstm_Model(self):

        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(self.date_size, self.feature_num)))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dropout(0, 2))
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        if self.CATEGORICAL:
            model.add(Dense(self.num_classes, activation='softmax'))
        else:
            model.add(Dense(1, activation='linear'))

        return model

    # Use get ~to3DArray
    def Load_Conv1D_Lstm_Model(self):

        model = Sequential()
        model.add(Conv1D(filters=32,
                         kernel_size=3,
                         strides=1,
                         input_shape=(date_size, feature_num)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv1D(filters=64, kernel_size=3, strides=1))
        model.add(LeakyReLU(alpha=0.2))
        model.add(LSTM(64, return_sequences=True, activation='tanh'))
        model.add(Dropout(0, 2))
        model.add(LSTM(32, return_sequences=False, activation='tanh'))
        model.add(Dropout(0, 2))
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        if self.CATEGORICAL:
            model.add(Dense(self.num_classes, activation='softmax'))
        else:
            model.add(Dense(1, activation='linear'))

        return model

    # Use get ~to3DArray
    def Load_Deep_Conv1D_Lstm_Model(self):
        print('You must Use Large Dataset!!!!!!!!!')
        model = Sequential()
        model.add(Conv1D(filters=32,
                         kernel_size=1,
                         strides=1,
                         input_shape=(self.date_size, self.feature_num)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv1D(filters=64, kernel_size=3, strides=1))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv1D(filters=128, kernel_size=1, strides=1))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv1D(filters=256, kernel_size=3, strides=1))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv1D(filters=512, kernel_size=1, strides=1))
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
        if self.CATEGORICAL:
            model.add(Dense(self.num_classes, activation='softmax'))
        else:
            model.add(Dense(1, activation='linear'))

        return model

    # Use get~to4DArray
    def Load_Conv2DTD_Lstm_Model(self):

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

        if self.CATEGORICAL:
            model.add(Dense(self.num_classes, activation='softmax'))
        else:
            model.add(Dense(1, activation='linear'))
        return model

    # Use get~to4DArray
    def Load_Conv2DTD_Lstm_IMG_Model(self):

        model = Sequential()
        model.add(TimeDistributed(Conv2D(32, (3, 3)), input_shape=(None, 38, 38, 1)))
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(LeakyReLU(alpha=0.2)))
        model.add(TimeDistributed(Conv2D(64, (1, 1))))
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(LeakyReLU(alpha=0.2)))
        model.add(TimeDistributed(Conv2D(128, (3, 3))))
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(LeakyReLU(alpha=0.2)))
        model.add(TimeDistributed(Conv2D(256, (1, 1))))
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(LeakyReLU(alpha=0.2)))
        model.add(TimeDistributed(Conv2D(512, (3, 3))))
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(LeakyReLU(alpha=0.2)))

        ###
        model.add(TimeDistributed(Flatten()))
        model.add(TimeDistributed(Dense(1024, activation='relu')))
        model.add(LSTM(256, return_sequences=True))

        model.add(TimeDistributed(Dense(1024, activation='relu')))
        model.add(TimeDistributed(Dropout(0.5)))

        model.add(LSTM(256, return_sequences=False))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))

        if self.CATEGORICAL:
            model.add(Dense(self.num_classes, activation='softmax'))
        else:
            model.add(Dense(1, activation='linear'))
        return model

    def Load_DenseNet121_Lstm_IMG_Model(self):
        base_model = DenseNet.DenseNet121(include_top=False, weights=None, input_shape=(38, 38, self.date_size))

        model = Sequential()
        model.add(base_model)
        # model.add(Dense(512))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Dense(256))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Dense(128))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Dense(64, activation='linear'))
        if self.CATEGORICAL:
            model.add(GlobalAveragePooling2D(name='avg_pool'))
            model.add(Dense(self.num_classes, activation='softmax'))
        else:
            model.add(Dense(1, activation='linear'))

        model.compile(loss='mse', optimizer='rmsprop', metrics=['acc'])

        return model

    def Load_Model(self):

        server, user, password, database = UtilStock.ParseConfig('config.ini')
        connect = mssql.connect(server=server, user=user, password=password, database=database, charset='UTF8')
        cur = connect.cursor()
        info = UtilStock.LoadFinanceStockInfo(cur)

        # LSTM /  CONV1D_LSTM / DEEP_CONV1D_LSTM / CONV2DTD_LSTM
        if self.MODEL_NAME == 'LSTM':
            data, label = datapreprocess.getFinanceInfoLabelto2DArray(cur, info, data_size=self.sample_size,
                                                                      date_size=self.date_size,
                                                                      scaler=self.SCALER, unit=self.UNIT, bLevel=self.CATEGORICAL)

            model = self.Load_Lstm_Model()
        elif self.MODEL_NAME == 'CONV1D_LSTM':
            data, label = datapreprocess.getFinanceInfoLabelto3DArray(cur, info, data_size=self.sample_size,
                                                                      date_size=self.date_size,
                                                                      scaler=self.SCALER, unit=self.UNIT, bLevel=self.CATEGORICAL)
            model = self.stock_model.Load_Conv1D_Lstm_Model()
        elif self.MODEL_NAME == 'DEEP_CONV1D_LSTM':
            data, label = datapreprocess.getFinanceInfoLabelto3DArray(cur, info, data_size=self.sample_size,
                                                                      date_size=self.date_size,
                                                                      scaler=self.SCALER, unit=self.UNIT, bLevel=self.CATEGORICAL)
            model = self.stock_model.Load_Deep_Conv1D_Lstm_Model()

        elif self.MODEL_NAME == 'CONV2DTD_LSTM':

            data, label = datapreprocess.getFinanceInfoLabelto4DArray(cur, info, data_size=self.sample_size,
                                                                      date_size=self.date_size,
                                                                      scaler=self.SCALER, unit=self.UNIT, bLevel=self.CATEGORICAL)
            model = self.stock_model.Load_Conv2DTD_Lstm_Model()

        elif self.MODEL_NAME == 'CONV2DTD_LSTM_IMG':

            data, label = datapreprocess.getFinanceInfoLabelto4DArray(cur, info, data_size=self.sample_size,
                                                                      date_size=self.date_size,
                                                                      scaler=self.SCALER, unit=self.UNIT, bLevel=self.CATEGORICAL,
                                                                      bImage=True)
            model = self.Load_Conv2DTD_Lstm_IMG_Model()

        elif self.MODEL_NAME == 'DENSE121_LSTM_IMG':
            data, label = datapreprocess.getFinanceInfotoImage(cur, info, data_size=self.sample_size, date_size=self.date_size,
                                                               scaler=self.SCALER, unit=self.UNIT, bLevel=self.CATEGORICAL)

            model = self.Load_DenseNet121_Lstm_IMG_Model()

        else:
            print('Is Not Exist Model')

        if CATEGORICAL:
            model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
        else:
            model.compile(loss='mse', optimizer='rmsprop', metrics=['acc'])
        print('Model Build...')
        model.summary()

        return data, label, model


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


if __name__ == "__main__":
    print('Use stock_train.py Or stock_test.py')
