import os
import math
import keras
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_squared_error
from stock_models import *

#################################################
#### Do it here

#data params
train_ratio = 0.9
feature_num = 19
sample_size = 10
date_size = 5
UNIT = 'WEEK'
SCALER = True
CATEGORICAL = True

#train params
train_epochs = 500

#model save path
MODEL_SAVE_FOLDER_PATH = './model/'
MODEL_NAME = 'LSTM'    # choose LSTM /  CONV1D_LSTM / DEEP_CONV1D_LSTM /  CONV2DTD_LSTM / DENSE121_LSTM / CONV2DTD_LSTM_IMG

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
              epochs=train_epochs,
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
    stock_model = STOCK_MODEL(feature_num,sample_size,date_size,UNIT,SCALER, CATEGORICAL, num_classes, MODEL_NAME)
    data, label, model = stock_model.Load_Model()

    train_model(model, data, label)