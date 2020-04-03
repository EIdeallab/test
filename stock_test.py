from datetime import datetime
from stock_models import *

model_path = './sota_bak/model_3.h5'  # CATECOREICAL_DEEP_CONV1D_LSTM37-0.0467.h5  /DEEP_CONV1D_LSTM65-0.0032.h5
code_num = 1000
CATEGORICAL = False
UNIT = 'WEEK'

if __name__ == "__main__":
    year = datetime.today().year
    month = datetime.today().month
    day = datetime.today().day

    #str_date = str(year) + '-' + str(month) + '-' + str(day)
    #for debug
    str_date = '2020-3-30'
    date_size = 5

    print(str_date)
    print('Check date...')

    model = load_model(model_path)

    if(CATEGORICAL):
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
    else:
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    server, user, password, database = UtilStock.ParseConfig('config.ini')
    connect = mssql.connect(server=server, user=user, password=password, database=database, charset='UTF8')
    cur = connect.cursor()
    info = UtilStock.LoadFinanceStockInfo(cur)


    print('data load....')
    newest_data, code = datapreprocess.getTestsetby3DArray(cur, info, str_date, date_size, unit = UNIT)
    print('data load done!')

    if(CATEGORICAL):
        testPredict = model.predict_classes(newest_data)
        predict = testPredict.ravel()
        predict_idx = predict.argsort()[-code_num:][::-1]
    else:
        testPredict = model.predict(newest_data)
        predict = testPredict.ravel()
        predict_idx = predict.argsort()[-code_num:][::-1]


    old_code = []

    file = open("CODE_predict.txt", 'w')

    for idx in predict_idx:
        if ((CATEGORICAL == True) and (predict[idx] > 6)) or ((CATEGORICAL == False) and (predict[idx] > 1)):
            predict_str = '추천 종목은' + str(code[idx]) + '입니다. 아마도' + str(predict[idx]) + '만큼 변동할것 입니다.'
            file.write(code[idx][0] + str(predict[idx]) + '\n')
            print(predict_str)

    file.close()

    print('Predict done!')