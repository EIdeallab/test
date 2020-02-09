import utilStock
import pymssql as mssql
import random
import numpy as np
import pandas as pd

def MinMaxScaler(data):
#정규화 함수
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)

def getInfoLabelto3DArray(cur, info, date_size, data_size = 0, scaler = False):
#주의 : list가 아닌 ndarray임

    rnd = []
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
        price = utilStock.LoadStockPriceByCode(cur, code.iloc[idx])
        price.drop('DATE', axis=1, inplace=True) #날짜 제거
        price = price.dropna()  #NONE값 가진 행 제거
        ratio = price['CHANGE_RATIO']
        price.drop('CHANGE_RATIO', axis=1, inplace=True)  # 날짜 제거
        dataset = price.as_matrix()

        if(scaler == True):
            dataset = MinMaxScaler(dataset)

        for j in range(0, len(dataset) - date_size):
            _x = dataset[j:j + date_size]
            _y = ratio[j + date_size: j + date_size + 1 ].values

            #reshape
            #_x = _x.reshape((1,_x.shape[0], _x.shape[1], 1))

            #첫 루프만
            if(i == 0 and j == 0):
                data  = _x
                label = _y
            else:
                data = np.concatenate([data,_x], -1)
                label = np.concatenate([label,_y])

    return data, label

def getInfoLabelto2DArray(cur, info, date_size, data_size = 0, scaler = False):
# data : LoadStockInfo 반환값, date_size : 몇 일씩 뭉칠껀지,
# data_size :주식코드 몇개에 대한 데이터를 만들껀지
# train list와 label list를 반환

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
        price = utilStock.LoadStockPriceByCode(cur, code.iloc[idx])
        print(price)

        price.drop('DATE', axis=1, inplace=True) #날짜 제거
        price = price.dropna()  #NONE값 가진 행 제거
        ratio = price['CHANGE_RATIO']
        price.drop('CHANGE_RATIO', axis=1, inplace=True)  # 날짜 제거
        dataset = price.as_matrix()

        if(scaler == True):
            dataset = MinMaxScaler(dataset)

        for i in range(0, len(dataset) - date_size):
            _x = dataset[i:i + date_size]
            _y = ratio[i + date_size: i + date_size + 1 ].values
            # print(i + seq_length+predict_day)
            data.append(_x)
            label.append(_y)

    return np.array(data), np.array(label)

def getFinanceInfoLabelto2DArray(cur, info, date_size, sample_size = 0, scaler = False):
# data : LoadStockInfo 반환값, date_size : 몇 일씩 뭉칠껀지,
# data_size :주식코드 몇개에 대한 데이터를 만들껀지
# train list와 label list를 반환

    data = []  # train data
    label = []  # label data

    if(sample_size == 0):
        sample_size = len(info)

    code = info['STOCK_CODE']
    for i in range(sample_size):

        #주식 코드별로 없는 데이터가 존재하므로 있을 때까지 반복해서 가져온다.
        while True :
            rnd_num = random.randint(0, len(info) - 1)
            price = utilStock.LoadStockFinanceByCode(cur, code.iloc[rnd_num])
            
            if price.empty == False :
                break
        
        price.drop('DATE', axis=1, inplace=True) #날짜 제거
        price = price.dropna()  #NONE값 가진 행 제거
        ratio = price['CHANGE_RATIO']
        price.drop('CHANGE_RATIO', axis=1, inplace=True)  # y값 제거
        dataset = price.as_matrix()

        if(scaler == True):
            dataset = MinMaxScaler(dataset)

        for i in range(0, len(dataset) - date_size):
            _x = dataset[i:i + date_size]
            _y = ratio[i + date_size: i + date_size + 1 ].values
            # print(i + seq_length+predict_day)
            data.append(_x)
            label.append(_y)

    return np.array(data), np.array(label)

#1DArray수정 필요
def getInfoLabelto1Dlist(cur, info, data_size = 0, scaler = False):

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
        price = utilStock.LoadStockPriceByCode(cur, code.iloc[idx])
        price.drop('DATE', axis=1, inplace=True) #날짜 제거
        price = price.dropna()  #NONE값 가진 행 제거
        ratio = price['CHANGE_RATIO']
        price.drop('CHANGE_RATIO', axis=1, inplace=True)  # 날짜 제거
        dataset = price.as_matrix()

        if(scaler == True):
            dataset = MinMaxScaler(dataset)

        dataset.T
        ratio.T
        data.append(dataset)
        label.append(ratio)

    return data, label

#연습장
if __name__ == "__main__":
    server, user, password, database = utilStock.ParseConfig('config.ini')
    connect = mssql.connect(server=server, user=user, password=password, database=database, charset='UTF8')
    cur = connect.cursor()
    info = utilStock.LoadStockInfo(cur)
    data, label = getInfoLabelto2DArray(cur, info, data_size=5, date_size=5,  scaler= True)
    print(1)





