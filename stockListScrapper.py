import pandas as pd
import pymssql as mssql
import time
import configparser
import utilStock

# 종목 정보 로드
code_df = pd.read_html('http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13', header=0)[0]

# 종목코드가 6자리이기 때문에 6자리를 맞춰주기 위해 설정해줌
code_df.종목코드 = code_df.종목코드.map('{:06d}'.format)

# 우리가 필요한 것은 회사명과 종목코드이기 때문에 필요없는 column들은 제외해준다.
code_df = code_df[['종목코드', '회사명', '업종']]

# 데이터베이스에 적재할 수 있도록 필드를 분리하고 컬럼 명을 바꿔준다.
name_df = code_df.rename(columns={'종목코드': 'STOCK_CODE', '회사명': 'STOCK_NAME', '업종' : 'STOCK_CATE'})
name_df.head()

# 컬럼을 데이터베이스에 적재하기 좋은 형태로 수정한다.
name_df = name_df[['STOCK_CODE', 'STOCK_NAME', 'STOCK_CATE']]

# mssql 서버 접속
sv, us, ps, db = utilStock.ParseConfig('config.ini')

conn = mssql.connect(server=sv, user=us, password=ps,database=db, charset= 'UTF8')
try:
    cur = conn.cursor()

    # 주식 기본정보 테이블 생성
    sql = "EXEC CREATE_TBL_STOCK_INFO"
    cur.execute(sql)

    # 로드한 기본 정보 데이터 mssql 업로드
    sql = "INSERT INTO STOCK_INFO VALUES(%s, %s, %s);"
    data = [tuple(x) for x in name_df.values]
    print (",,, sql_statement", "," * 100, "\n")
    cur.executemany(sql, data)

    # 주식 종목 분류정보 테이블 생성
    sql = "EXEC CREATE_TBL_STOCK_CATE"
    cur.execute(sql)

    conn.commit()

finally:
    conn.close()