import pymssql as mssql
import UtilStock
import stockListScrapper as sls
import stockPriceScrapper as sps
import financeScrapper as fs
import themaScrapper as ts
import pandas as pd
import datetime as dt

# mssql 서버 접속 전 반드시 SP를 먼저 등록해야한다.

# mssql 서버 접속
sv, us, ps, db = UtilStock.ParseConfig('config.ini')

conn = mssql.connect(server=sv, user=us, password=ps,database=db, charset= 'UTF8')
try:
    cur = conn.cursor()

    f = open("CODE_predict.txt", 'r')
    stock_list = []
    while True:
        line = f.readline()
        if not line: break
        code_ratio = line.split(' ')
        stock_list.append((code_ratio[0],code_ratio[1]))
    f.close()

    # 일부 데이터만 로드
    f = open("CODE_review.txt", 'w')
    for (stock_code, ratio) in stock_list:
        url = 'http://finance.naver.com/item/sise_day.nhn?code={code}'.format(code=stock_code)
        url = url.strip()
        print("Load Stock info : ", url)
    
        # 최신 일자 데이터 가져오기
        price_df = pd.DataFrame() 

        # Url에서 데이터 수집
        pg_url = '{url}&page=1'.format(url=url)
        price_df = price_df.append(pd.read_html(pg_url, header=0)[0], ignore_index=True)
            
        # df.dropna()를 이용해 결측값 있는 행 제거 
        price_df = price_df.dropna() 
        
        # 한글로 된 컬럼명을 영어로 바꿔줌 
        price_df = price_df.rename(
            columns= {
                '날짜': 'DATE', '종가': 'CLOSE', '전일비': 'DIFF', '시가': 'OPEN', 
                '고가': 'HIGH', '저가': 'LOW', '거래량': 'VOLUME'}) 

        # 데이터의 타입을 int형으로 바꿔줌 
        price_df[['CLOSE','DIFF','OPEN','HIGH','LOW','VOLUME']] = price_df[['CLOSE','DIFF','OPEN','HIGH','LOW','VOLUME']].astype(int) 
        
        # 컬럼명 'date'의 타입을 date로 바꿔줌 
        price_df['DATE'] = pd.to_datetime(price_df['DATE']) 

        updated = False
        # 얻어낸 컬럼의 각각의 row에 대해서 처리

        weekday = ['MON','TUE','WED','THU','FRI','SAT','SUN']
        for index, row in price_df.iterrows():
            # 최신데이터
            if index == 1 :
                curr_price = row['CLOSE']
                curr_date = row['DATE']
            # 이전데이터
            else :
                if weekday[row['DATE'].weekday()] == 'FRI' :
                    last_price = row['CLOSE']
                    last_date = row['DATE']
        
        curr_date_str = '%3s' % weekday[curr_date.weekday()]
        last_date_str = '%3s' % weekday[last_date.weekday()]
        curr_price_str = '%8s' % str(curr_price)
        last_price_str = '%8s' % str(last_price)
        pred_ratio_str = '%6s' % str(round(float(ratio), 2))
        true_ratio_str = '%6s' % str(round(float(curr_price)/last_price, 2))

        f.write(stock_code + ' ' + 
                last_date_str + ' ' + last_price_str + ' ' + 
                curr_date_str + ' ' + curr_price_str + ' ' + 
                pred_ratio_str + ' ' + true_ratio_str +'\n') 

        print('Check Stock Code : ' + stock_code)
    f.close()

        

    # 일부 데이터에 대해서 

    conn.commit()

    print('TABLE UPDATE COMPLETE')

finally:
    conn.close()