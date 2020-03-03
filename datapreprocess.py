import UtilStock
import pymssql as mssql
import random
import numpy as np
import pandas as pd
import copy
import cv2

def makeDatatoImage(_dataset):

    Image = []
    index = np.array(((0 ,1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18),
                  (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 0),
                  (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 0, 1),
                  (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 0, 1, 2),
                  (4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 0, 1, 2, 3),
                  (5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 0, 1, 2, 3, 4),
                  (6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 0, 1, 2, 3, 4, 5),
                  (7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 0, 1, 2, 3, 4, 5, 6),
                  (8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 0, 1, 2, 3, 4, 5, 6, 7),
                  (9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 0, 1, 2, 3, 4, 5, 6, 7, 8),
                  (10, 11, 12, 13, 14, 15, 16, 17, 18, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
                  (11, 12, 13, 14, 15, 16, 17, 18, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
                  (12, 13, 14, 15, 16, 17, 18, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
                  (13, 14, 15, 16, 17, 18, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
                  (14, 15, 16, 17, 18, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13),
                  (15, 16, 17, 18, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14),
                  (16, 17, 18, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
                  (17, 18, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16),
                  (18, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17),))

    for data in _dataset:
        tmp = []
        for idx in index:
            tmp.append(copy.deepcopy(data[idx]))

        tmp = np.array(tmp)
        flipimage1 = cv2.flip(tmp, 0)
        flipimage2 = cv2.flip(tmp, 1)
        flipimage3 = cv2.flip(flipimage1, 1)

        addv1 = cv2.vconcat([tmp, flipimage3])
        addv2 = cv2.vconcat([flipimage1, flipimage2])
        addImage = cv2.hconcat([addv1, addv2])

        Image.append(addImage)

    return np.array(Image)

def getLevel(values,unit = 'DAY'):

    #12levels

    # if (unit == 'DAY'):
    #     if (values < 0.9):
    #         _y = [0]
    #     elif (values >= 0.9) and (values < 0.92):
    #         _y = [1]
    #     elif (values >= 0.92) and (values < 0.94):
    #         _y = [2]
    #     elif (values >= 0.94) and (values < 0.96):
    #         _y = [3]
    #     elif (values >= 0.96) and (values < 0.98):
    #         _y = [4]
    #     elif (values >= 0.98) and (values < 1):
    #         _y = [5]
    #     elif (values >= 1) and (values < 1.02):
    #         _y = [6]
    #     elif (values >= 1.02) and (values < 1.04):
    #         _y = [7]
    #     elif (values >= 1.04) and (values < 1.06):
    #         _y = [8]
    #     elif (values >= 1.06) and (values < 1.08):
    #         _y = [9]
    #     elif (values >= 1.08) and (values < 1.1):
    #         _y = [10]
    #     elif (values >= 1.1):
    #         _y = [11]
    #
    # elif (unit == 'WEEK'):
    #     if (values < 0.7):
    #         _y = [0]
    #     elif (values >= 0.7) and (values < 0.75):
    #         _y = [1]
    #     elif (values >= 0.75) and (values < 0.8):
    #         _y = [2]
    #     elif (values >= 0.8) and (values < 0.85):
    #         _y = [3]
    #     elif (values >= 0.85) and (values < 0.9):
    #         _y = [4]
    #     elif (values >= 0.9) and (values < 0.95):
    #         _y = [5]
    #     elif (values >= 0.95) and (values < 1):
    #         _y = [6]
    #     elif (values >= 1) and (values < 1.05):
    #         _y = [7]
    #     elif (values >= 1.05) and (values < 1.1):
    #         _y = [8]
    #     elif (values >= 1.1) and (values < 1.15):
    #         _y = [9]
    #     elif (values >= 1.15) and (values < 1.2):
    #         _y = [10]
    #     elif (values >= 1.2) and (values < 1.25):
    #         _y = [11]
    #     elif (values >= 1.25) and (values < 1.3):
    #         _y = [12]
    #     elif (values >= 1.3):
    #         _y = [13]
    #
    # elif (unit == 'MONTH'):
    #     print('Not yet :) ')


    #4levels

    # if (unit == 'DAY'):
    #     if (values < 0.98):
    #         _y = [0]
    #     elif (values >= 0.98) and (values < 1):
    #         _y = [1]
    #     elif (values >= 1) and (values < 1.02):
    #         _y = [2]
    #     elif (values >= 1.02):
    #         _y = [3]
    #
    # elif (unit == 'WEEK'):
    #     if (values < 0.95):
    #         _y = [0]
    #     elif (values >= 0.95) and (values < 1):
    #         _y = [1]
    #     elif (values >= 1) and (values < 1.05):
    #         _y = [2]
    #     elif (values >= 1.05):
    #         _y = [3]
    # elif (unit == 'MONTH'):
    #     print('Not yet :) ')



    if (unit == 'DAY'):
        if (values < 1):
            _y = [0]
        elif (values >= 1):
            _y = [1]

    elif (unit == 'WEEK'):
        if (values < 1):
            _y = [0]
        elif (values >= 1):
            _y = [1]

    elif (unit == 'MONTH'):
        print('Not yet :) ')

    return _y

def MinMaxScaler(data):
#정규화 함수
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)

def getInfoLabelto3DArray(cur, info, date_size, data_size = 0, scaler = False, unit = 'DAY', bLevel = False):
#주의 : list가 아닌 ndarray임
# unit = 'WEEK', 'MONTH', 'DAY'
    rnd = []
    data =[]
    label = []
    if(data_size == 0):
        data_size = len(info)

    rnd_num = random.randint(0 ,len(info))

    #랜덤하게 갖고 오고 싶었음
    for i in range(data_size):
        while rnd_num in rnd:
            rnd_num = random.randint(0, len(info) - 1)
        rnd.append(rnd_num)

    code = info['STOCK_CODE']
    for i in range(data_size):
        idx   = rnd[i]

        if (unit == 'DAY'):
            price = UtilStock.LoadStockPriceByCode(cur, code.iloc[idx])
        elif (unit == 'WEEK'):
            print('Not yet :) ')
            break
        elif (unit == 'MONTH'):
            print('Not yet :) ')
            break

        price.drop('DATE', axis=1, inplace=True) #날짜 제거
        price = price.dropna()  #NONE값 가진 행 제거
        ratio = price['CHANGE_RATIO']
        price.drop('CHANGE_RATIO', axis=1, inplace=True)  # 날짜 제거
        dataset = price.as_matrix()

        if(scaler == True):
            dataset = MinMaxScaler(dataset)

        for j in range(0, len(dataset) - date_size):
            _x = dataset[j:j + date_size]

            if(bLevel):
                _y = getLevel(ratio[j + date_size: j + date_size + 1].values, unit)
            else:
                _y = ratio[j + date_size: j + date_size + 1 ].values

            #reshape
            _x = _x.reshape((1, _x.shape[0], _x.shape[1]))

            data.append(_x[0].tolist())
            label.append(_y[0])

    return np.array(data), np.array(label)

def getInfoLabelto2DArray(cur, info, date_size, data_size = 0, scaler = False, unit = 'DAY', bLevel = False):
# data : LoadStockInfo 반환값, date_size : 몇 일씩 뭉칠껀지,
# data_size :주식코드 몇개에 대한 데이터를 만들껀지
# train list와 label list를 반환
# unit = 'WEEK', 'MONTH', 'DAY'
    rnd = []
    data = []  # train data
    label = []  # label data

    if(data_size == 0):
        data_size = len(info)

    rnd_num = random.randint(0 ,len(info))

    #랜덤하게 갖고 오고 싶었음
    for i in range(data_size):
        while rnd_num in rnd:
            rnd_num = random.randint(0, len(info) - 1)
        rnd.append(rnd_num)

    code = info['STOCK_CODE']
    for i in range(data_size):
        idx   = rnd[i]

        if (unit == 'DAY'):
            price = UtilStock.LoadStockPriceByCode(cur, code.iloc[idx])
        elif (unit == 'WEEK'):
            print('Not yet :) ')
            break
        elif (unit == 'MONTH'):
            print('Not yet :) ')
            break

        price.drop('DATE', axis=1, inplace=True) #날짜 제거
        price = price.dropna()  #NONE값 가진 행 제거
        ratio = price['CHANGE_RATIO']
        price.drop('CHANGE_RATIO', axis=1, inplace=True)  # 날짜 제거
        dataset = price.as_matrix()

        if(scaler == True):
            dataset = MinMaxScaler(dataset)

        for i in range(0, len(dataset) - date_size):
            _x = dataset[i:i + date_size]

            if(bLevel):
                _y = getLevel(ratio[j + date_size: j + date_size + 1].values, unit)
            else:
                _y = ratio[i + date_size: i + date_size + 1 ].values
            # print(i + seq_length+predict_day)
            data.append(_x)
            label.append(_y)

    return np.array(data), np.array(label)

def getFinanceInfoLabelto2DArray(cur, info, date_size, data_size=0, scaler=False, unit = 'DAY', bLevel = False):
    # data : LoadStockInfo 반환값, date_size : 몇 일씩 뭉칠껀지,
    # data_size :주식코드 몇개에 대한 데이터를 만들껀지
    # train list와 label list를 반환
    # unit = 'WEEK', 'MONTH', 'DAY'

    rnd = []
    data = []  # train data
    label = []  # label data

    if (data_size == 0):
        data_size = len(info)

    rnd_num = random.randint(0 ,len(info))

    #랜덤하게 갖고 오고 싶었음
    for i in range(data_size):
        while rnd_num in rnd :
            rnd_num = random.randint(0, len(info) - 1)
        rnd.append(rnd_num)

    code = info['STOCK_CODE']
    for i in range(data_size):
        idx   = rnd[i]

        if(unit == 'DAY'):
            price = UtilStock.LoadStockFinanceByCode(cur, code.iloc[idx])
        elif (unit == 'WEEK'):
            price = UtilStock.LoadStockFinanceWeekByCode(cur, code.iloc[idx])
        elif (unit == 'MONTH'):
            print('Not yet :) ')
            break

        price.drop('DATE', axis=1, inplace=True)  # 날짜 제거
        price = price.dropna()  # NONE값 가진 행 제거
        price = price.apply(pd.to_numeric, downcast='float')
        ratio = price['CHANGE_RATIO']
        price.drop('CHANGE_RATIO', axis=1, inplace=True)  # y값 제거
        dataset = price.as_matrix()

        if (scaler == True):
            dataset = MinMaxScaler(dataset)

        for j in range(0, len(dataset) - date_size):

            _x = dataset[j:j + date_size]
            if(bLevel):
                _y = getLevel(ratio[j + date_size: j + date_size + 1].values, unit)
            else:
                _y = ratio[j + date_size: j + date_size + 1].values
            # print(i + seq_length+predict_day)
            data.append(_x)
            label.append(_y)

    return np.array(data), np.array(label)

def getFinanceInfoLabelto3DArray(cur, info, date_size, data_size=0, scaler=False, unit = 'DAY', bLevel = False):
    # unit = 'WEEK', 'MONTH', 'DAY'

    rnd = []
    data = []
    label = []

    if (data_size == 0):
        data_size = len(info)


    rnd_num = random.randint(0 ,len(info))

    #랜덤하게 갖고 오고 싶었음
    for i in range(data_size):
        while rnd_num in rnd :
            rnd_num = random.randint(0, len(info) - 1)
        rnd.append(rnd_num)

    code = info['STOCK_CODE']
    for i in range(data_size):

        idx   = rnd[i]

        if (unit == 'DAY'):
            price = UtilStock.LoadStockFinanceByCode(cur, code.iloc[idx])
        elif (unit == 'WEEK'):
            price = UtilStock.LoadStockFinanceWeekByCode(cur, code.iloc[idx])
        elif (unit == 'MONTH'):
            print('Not yet :) ')
            break

        price.drop('DATE', axis=1, inplace=True)  # 날짜 제거
        price = price.dropna()  # NONE값 가진 행 제거
        ratio = price['CHANGE_RATIO']
        price.drop('CHANGE_RATIO', axis=1, inplace=True)  # y값 제거
        dataset = price.as_matrix()

        if (scaler == True):
            dataset = MinMaxScaler(dataset)

        for j in range(0, len(dataset) - date_size):
            _x = dataset[j:j + date_size]

            if(bLevel):
                _y = getLevel(ratio[j + date_size: j + date_size + 1].values,unit)
            else:
                _y = ratio[j + date_size: j + date_size + 1].values
            #reshape
            _x = _x.reshape((1, _x.shape[0], _x.shape[1]))

            data.append(_x[0].tolist())
            label.append(_y[0])

    return np.array(data), np.array(label)

def getFinanceInfoLabelto4DArray(cur, info, date_size, data_size=0, scaler=False, unit = 'DAY', labelunit = False,  bLevel = False, bImage = False):
    # unit = 'WEEK', 'MONTH', 'DAY'

    rnd = []
    data = []
    label = []

    if (data_size == 0):
        data_size = len(info)

    rnd_num = random.randint(0 ,len(info))

    #랜덤하게 갖고 오고 싶었음
    for i in range(data_size):
        while rnd_num in rnd :
            rnd_num = random.randint(0, len(info) - 1)
        rnd.append(rnd_num)

    code = info['STOCK_CODE']
    for i in range(data_size):

        idx   = rnd[i]

        if (unit == 'DAY'):
            price = UtilStock.LoadStockFinanceByCode(cur, code.iloc[idx])
        elif (unit == 'WEEK'):
            price = UtilStock.LoadStockFinanceWeekByCode(cur, code.iloc[idx])
        elif (unit == 'MONTH'):
            print('Not yet :) ')
            break

        price.drop('DATE', axis=1, inplace=True)  # 날짜 제거
        price = price.dropna()  # NONE값 가진 행 제거
        ratio = price['CHANGE_RATIO']
        price.drop('CHANGE_RATIO', axis=1, inplace=True)  # y값 제거
        if not(bImage):
            price.drop('MARKET_CAP', axis=1, inplace=True) #임시
        dataset = price.as_matrix()

        if (scaler == True):
            dataset = MinMaxScaler(dataset)

        for j in range(0, len(dataset) - date_size):

            if labelunit:
                _x = dataset[j:j + date_size]
                _y = ratio[j + 1: j + date_size + 1].values
                _x = _x.reshape((1, _x.shape[0], 3, 6, 1))  ###
                _y = _y.reshape((1, _y.shape[0], 1))

            else:
                _x = dataset[j:j + date_size]
                if(bImage):
                    _x = makeDatatoImage(_x)
                    _x = _x.reshape((1, _x.shape[0], 38, 38, 1))  ########하드코딩
                else:
                    _x = _x.reshape((1, _x.shape[0], 3, 6, 1))  ########하드코딩
                if(bLevel):
                    _y = getLevel(ratio[j + date_size: j + date_size + 1].values, unit)
                else:
                    _y = ratio[j + date_size: j + date_size + 1].values

            data.append(_x[0].tolist())
            label.append(_y[0])

    return np.array(data), np.array(label)


def getFinanceInfotoImage(cur, info, date_size, data_size=0, scaler=False, unit = 'DAY', bLevel = False):
    # unit = 'WEEK', 'MONTH', 'DAY'

    rnd = []
    data = []
    label = []

    if (data_size == 0):
        data_size = len(info)

    rnd_num = random.randint(0 ,len(info))

    #랜덤하게 갖고 오고 싶었음
    for i in range(data_size):
        while rnd_num in rnd :
            rnd_num = random.randint(0, len(info) - 1)
        rnd.append(rnd_num)

    code = info['STOCK_CODE']
    for i in range(data_size):

        idx   = rnd[i]

        if (unit == 'DAY'):
            price = UtilStock.LoadStockFinanceByCode(cur, code.iloc[idx])
        elif (unit == 'WEEK'):
            price = UtilStock.LoadStockFinanceWeekByCode(cur, code.iloc[idx])
        elif (unit == 'MONTH'):
            print('Not yet :) ')
            break

        price.drop('DATE', axis=1, inplace=True)  # 날짜 제거
        price = price.dropna()  # NONE값 가진 행 제거
        ratio = price['CHANGE_RATIO']
        price.drop('CHANGE_RATIO', axis=1, inplace=True)  # y값 제거
        dataset = price.as_matrix()

        if (scaler == True):
            dataset = MinMaxScaler(dataset)

        for j in range(0, len(dataset) - date_size):

            _x = dataset[j:j + date_size]
            _x = makeDatatoImage(_x)
            _x = _x.reshape((38, 38, _x.shape[0]))  ########하드코딩

            if (bLevel):
                _y = getLevel(ratio[j + date_size: j + date_size + 1].values, unit)
            else:
                _y = ratio[j + date_size: j + date_size + 1].values

            data.append(_x.tolist())
            label.append(_y[0])

    return np.array(data), np.array(label)

#1DArray수정 필요
def getInfoLabelto1Dlist(cur, info, data_size = 0, scaler = False, unit = 'DAY'):
    # unit = 'WEEK', 'MONTH', 'DAY'

    rnd = []
    data = []  # train data
    label = []  # label data

    if(data_size == 0):
        data_size = len(info)

    rnd_num = random.randint(0 ,len(info))

    #랜덤하게 갖고 오고 싶었음
    for i in range(data_size):
        while rnd_num in rnd:
            rnd_num = random.randint(0, len(info) - 1)
        rnd.append(rnd_num)

    code = info['STOCK_CODE']
    for i in range(data_size):
        idx   = rnd[i]

        if (unit == 'DAY'):
            price = UtilStock.LoadStockPriceByCode(cur, code.iloc[idx])
        elif (unit == 'WEEK'):
            print('Not yet :) ')
            break
        elif (unit == 'MONTH'):
            print('Not yet :) ')
            break

        price.drop('DATE', axis=1, inplace=True) #날짜 제거
        price = price.dropna()  #NONE값 가진 행 제거
        ratio = price['CHANGE_RATIO']
        price.drop('CHANGE_RATIO', axis=1, inplace=True)  # 날짜 제거
        dataset = price.as_matrix()

        if(scaler == True):
            test_min = np.min(dataset, 0)
            test_max = np.max(dataset, 0)
            test_denom = test_max - test_min
            dataset = MinMaxScaler(dataset)

        dataset.T
        ratio.T
        data.append(dataset)
        label.append(ratio)

    return data, label

#연습장
if __name__ == "__main__":
    server, user, password, database = UtilStock.ParseConfig('config.ini')
    connect = mssql.connect(server=server, user=user, password=password, database=database, charset='UTF8')
    cur = connect.cursor()
    info = UtilStock.LoadStockInfo(cur)
    data, label = getFinanceInfoLabelto4DArray(cur, info, data_size=0, date_size=5,  scaler= True)
    print(1)






