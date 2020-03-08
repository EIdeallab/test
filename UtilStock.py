import pymssql as mssql
import pandas as pd
import configparser


#config parser
def ParseConfig(file_name):
    config = configparser.ConfigParser()
    config.read(file_name)

    return config['DATABASE_CONFIG']['HOST'], config['DATABASE_CONFIG']['USER'], \
    config['DATABASE_CONFIG']['PASSWORD'], config['DATABASE_CONFIG']['DATABASE']

#StockInfo반환 함수
def LoadStockInfo(cur):
    sql = "SELECT * FROM STOCK_INFO"
    print("sql:", sql)
    cur.execute(sql)
    info_col = ('STOCK_CODE','STOCK_NAME','STOCK_CATE')
    Info = pd.DataFrame(cur.fetchall(),  columns = info_col )

    return Info

#재무 정보가 있는 StockInfo반환 함수
def LoadFinanceStockInfo(cur):
    sql = "SELECT * FROM STOCK_INFO WHERE STOCK_CODE IN (SELECT STOCK_CODE FROM STOCK_TRAINING_SET_WEEK)"
    print("sql:", sql)
    cur.execute(sql)
    info_col = ('STOCK_CODE','STOCK_NAME','STOCK_CATE')
    Info = pd.DataFrame(cur.fetchall(),  columns = info_col )

    return Info

#주식 EMBEDDING_SET 반환 함수
def LoadStockEmbeddingSet(cur):
    sql = "SELECT * FROM STOCK_EMBEDDING_SET"
    print("sql:", sql)
    cur.execute(sql)
    info_col = ('BEF_STOCK','AFT_STOCK')
    data = pd.DataFrame(cur.fetchall(),  columns = info_col )

    sql = "SELECT * FROM STOCK_FALSE_SET"
    print("sql:", sql)
    cur.execute(sql)
    info_col = ('BEF_STOCK','AFT_STOCK')
    false_data = pd.DataFrame(cur.fetchall(),  columns = info_col )

    sql = "SELECT COUNT(DISTINCT BEF_STOCK) BEF_CNT, COUNT(DISTINCT AFT_STOCK) AFT_CNT FROM STOCK_EMBEDDING_SET"
    print("sql:", sql)
    cur.execute(sql)
    info_col = ('BEF_CNT','AFT_CNT')
    data_cnt = pd.DataFrame(cur.fetchall(),  columns = info_col )

    return data, false_data, data_cnt

#특정 주식 코드의 데이터 반환 함수, 날짜 오름차순
def LoadStockPriceByCode(cur, StockCode):
    sql = "SEL_STOCK_PRICE " +StockCode
    print("sql:", sql)
    cur.execute(sql)
    price_col = ('AVERAGE','HIGHEST','LOWEST','VOLUME','DATE','CHANGE_RATIO')
    Info = pd.DataFrame(cur.fetchall(), columns =price_col )

    return Info

#특정 주식 코드의 재무재표 포함 반환 함수, 날짜 오름차순
def LoadStockFinanceByCode(cur, StockCode):
    sql = "SEL_STOCK_TRAINING_DATA " +StockCode
    print("sql:", sql)
    cur.execute(sql)
    price_col = (
        'STOCK_CODE', 'AVERAGE', 'HIGHEST', 'LOWEST', 'VOLUME', 'CHANGE_RATIO', 
        'MARKET_CAP', 'PBR', 'PBR2', 'PDR', 'PER', 'PCR_OP', 'PCR_IV', 'PCR_FI', 
        'PSR', 'PFR', 'PXR', 'ROE', 'ROA', 'MARKET_RANK', 'DATE')
    Info = pd.DataFrame(cur.fetchall(), columns =price_col )
    return Info

#특정 주식 코드의 재무재표 포함 반환 함수(주단위), 날짜 오름차순
def LoadStockFinanceWeekByCode(cur, StockCode):
    sql = "SEL_STOCK_TRAINING_DATA_WEEK " + StockCode
    print("sql:", sql)
    cur.execute(sql)
    price_col = (
        'STOCK_CODE', 'AVERAGE', 'HIGHEST', 'LOWEST', 'VOLUME', 'CHANGE_RATIO', 
        'MARKET_CAP', 'PBR', 'PBR2', 'PDR', 'PER', 'PCR_OP', 'PCR_IV', 'PCR_FI', 
        'PSR', 'PFR', 'PXR', 'ROE', 'ROA', 'MARKET_RANK', 'DATE')
    Info = pd.DataFrame(cur.fetchall(), columns =price_col )
    return Info

#특정 주식 코드의 재무재표 포함 테스트셋 반환 함수, 날짜 오름차순
def LoadStockTestsetByCode(cur, stockcode, date):
    sql = "SEL_STOCK_TEST_DATA " +stockcode + ',' + date
    print("sql:", sql)
    cur.execute(sql)
    price_col = (
        'STOCK_CODE', 'AVERAGE', 'HIGHEST', 'LOWEST', 'VOLUME', 'CHANGE_RATIO', 
        'MARKET_CAP', 'PBR', 'PBR2', 'PDR', 'PER', 'PCR_OP', 'PCR_IV', 'PCR_FI', 
        'PSR', 'PFR', 'PXR', 'ROE', 'ROA', 'MARKET_RANK', 'DATE')
    Info = pd.DataFrame(cur.fetchall(), columns =price_col )
    return Info

#특정 주식 코드의 재무재표 포함 테스트셋 반환 함수(주단위), 날짜 오름차순
def LoadStockTestsetWeekByCode(cur, stockcode, date):
    sql = "SEL_STOCK_TEST_DATA_WEEK " + stockcode + ',' + date
    print("sql:", sql)
    cur.execute(sql)
    price_col = (
        'STOCK_CODE', 'AVERAGE', 'HIGHEST', 'LOWEST', 'VOLUME', 'CHANGE_RATIO', 
        'MARKET_CAP', 'PBR', 'PBR2', 'PDR', 'PER', 'PCR_OP', 'PCR_IV', 'PCR_FI', 
        'PSR', 'PFR', 'PXR', 'ROE', 'ROA', 'MARKET_RANK', 'DATE')
    Info = pd.DataFrame(cur.fetchall(), columns =price_col )
    return Info

#주식 가격 전부 불러오기
def LoadStockPrice(cur):
    sql = "SELECT * FROM STOCK_PRICE ORDER BY DATE"
    print("sql:", sql)
    cur.execute(sql)
    price_col = ('STOCK_CODE','START','CLOSE','HIGHEST','LOWEST','VOLUME','DATE')
    Info = pd.DataFrame(cur.fetchall(), columns =price_col )

    return Info

def ProcessFinance(fn_cells):
    # 데이터프레임 가공
    raw_dts = pd.read_html(str(fn_cells[3]))
    raw_dt = raw_dts[0]
    keyValue = raw_dt['주요재무정보']
    serialData = raw_dt['분기']
    serialData.rows = keyValue
    df = pd.concat([keyValue, serialData], axis=1)
    df = df.set_index('주요재무정보')
    df = df.dropna(how='all', axis=1)
    columns = [s for s in df.columns if "(E)" not in s]
    df=df[columns]
    df.columns = df.columns.str.replace("IFRS연결","")
    df.columns = df.columns.str.replace("IFRS별도","")
    df.columns = df.columns.str.replace("GAAP개별","")
    df.columns = df.columns.str.replace("(","")
    df.columns = df.columns.str.replace(")","")
    df.columns = df.columns.str.replace("/","-")
    df.columns = df.columns.str.strip() + "-01"
    return df.fillna(0)

#연습장
if __name__ == "__main__":
    server, user, password, database = ParseConfig('config.ini')
    connect = mssql.connect(server=server, user=user, password=password, database=database, charset='UTF8')
    cur = connect.cursor()
   # info_data = LoadStockInfo(cur)
   # price_data = LoadStockPrice(cur,info_data[0][0])
    Info = LoadStockPrice(cur)
    print(1)


