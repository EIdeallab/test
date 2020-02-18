import pandas as pd
import pymssql as mssql
import time
import configparser
import UtilStock

def stockPriceScrap():
    # mssql 서버 접속
    sv, us, ps, db = UtilStock.ParseConfig('config.ini')

    conn = mssql.connect(server=sv, user=us, password=ps,database=db, charset= 'UTF8')
    try:
        cur = conn.cursor()

        # 주식 가격정보 테이블 생성
        sql = "EXEC CREATE_TBL_STOCK_PRICE"
        cur.execute(sql)
        conn.commit()

        # Sql 서버에서 주식 종목 리스트 로드
        sql = "SELECT STOCK_CODE FROM STOCK_INFO;"
        print("sql:", sql)
        cur.execute(sql)
        name_df =  pd.DataFrame(cur.fetchall())

        # 네이버 주식 데이터 가져오기
        data = [item[0] for item in name_df.values]
        for stockCode in data:
            url = 'http://finance.naver.com/item/sise_day.nhn?code={code}'.format(code=stockCode)
            url = url.strip()
            print("Load Stock info : ", url)
        
            # DB에 저장되지 않은 최신 일자 데이터 가져오기
            for page in range(1, 32):
                
                # 일자 데이터를 담을 df라는 DataFrame 정의 
                price_df = pd.DataFrame() 

                # Url에서 데이터 수집
                pg_url = '{url}&page={page}'.format(url=url, page=page)
                price_df = price_df.append(pd.read_html(pg_url, header=0)[0], ignore_index=True) 
                price_df['CODE'] = stockCode
                    
                # df.dropna()를 이용해 결측값 있는 행 제거 
                price_df = price_df.dropna() 
                
                # 한글로 된 컬럼명을 영어로 바꿔줌 
                price_df = price_df.rename(
                    columns= {
                        '날짜': 'DATE', '종가': 'CLOSE', '전일비': 'DIFF', '시가': 'OPEN', 
                        '고가': 'HIGH', '저가': 'LOW', '거래량': 'VOLUME'}) 

                # 데이터의 타입을 int형으로 바꿔줌 
                price_df[['CLOSE','DIFF','OPEN','HIGH','LOW','VOLUME']] = price_df[['CLOSE','DIFF','OPEN','HIGH','LOW','VOLUME']].astype(int) 
                
                # 컬럼을 데이터베이스에 적재하기 좋은 형태로 수정
                price_df = price_df[['CODE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'DIFF', 'VOLUME', 'DATE']]
                    
                # 컬럼명 'date'의 타입을 date로 바꿔줌 
                price_df['DATE'] = pd.to_datetime(price_df['DATE']) 

                updated = False
                # 얻어낸 컬럼의 각각의 row에 대해서 처리
                for index, row in price_df.iterrows():
                    # 얻은 데이터가 DB에 없는 경우 업로드
                    sql = "DECLARE @rtn int "
                    sql += "EXEC @rtn = UPDATE_STOCK_PRICE {0}, {1}, {2}, {3}, {4}, {5}, '{6}' SELECT @rtn".format(
                        stockCode, row['OPEN'], row['CLOSE'], row['HIGH'], row['LOW'], row['VOLUME'], row['DATE']
                    )
                    cur.execute(sql)
                    rt = cur.fetchall()
                    if rt[0][0] == -1 : 
                        updated = True
                        break
                # 요청 후 대기(트래픽 과다 방지)
                conn.commit()
                time.sleep(0.05)
                price_df.iloc[0:0]
                
                # 업데이트 완료시 다음 주식 정보 로드
                if updated == True :
                    break
                
            print('Update Stock info... {0}'.format(stockCode))
        
        # 주식 전일대비 증감치 컬럼 생성(결과값)
        sql = "EXEC UPDATE_STOCK_PRICE_RATIO"
        cur.execute(sql)
        conn.commit()
            
    finally:
        conn.close()