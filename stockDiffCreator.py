import pandas as pd
import LoadStock
import pymssql as mssql


#어제대비 증감률을 반환
#input은 LoadStockPrice의 반환값
def calcIncrease(data):

    average = (data['START'] + data['CLOSE']) / 2

    Increase = []
    for i in range(average.size - 1):
        # 이전 데이터 대비 증감률
        Increase.append((average.iloc[i + 1] - average.iloc[i]) / average.iloc[i])

    return Increase

def addIncreaseInfo(data):

    for i in range(data.size):
        price = LoadStock.LoadStockPriceByCode(cur, data['STOCK_CODE'][i])
        increase = calcIncrease(price)
        price = price.drop([0]) #가장 옛날 날짜 제거
        price['INCREASE'] = increase
        if(i == 0):
            info = price
        else:
            info = pd.concat([info, price])

    return info

#연습장
if __name__ == "__main__":
    server, user, password, database = LoadStock.ParseConfig('config.ini')
    connect = mssql.connect(server=server, user=user, password=password, database=database, charset='UTF8')
    cur = connect.cursor()
    info = LoadStock.LoadStockInfo(cur)
    newInfo = addIncreaseInfo(info)
    print(1)





