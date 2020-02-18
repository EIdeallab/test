import pymssql as mssql
import UtilStock
import stockListScrapper as sls
import stockPriceScrapper as sps
import financeScrapper as fs
import themaScrapper as ts

# mssql 서버 접속 전 반드시 SP를 먼저 등록해야한다.

# mssql 서버 접속
sv, us, ps, db = UtilStock.ParseConfig('config.ini')

conn = mssql.connect(server=sv, user=us, password=ps,database=db, charset= 'UTF8')
try:
    cur = conn.cursor()
    
    sls.stockListScrap()
    sps.stockPriceScrap()
    fs.financeScrap()
    #ts.themeScrap()

    # 파생 테이블 생성
    sql = 'EXEC CREATE_DRV_TABLES'
    
    cur.execute(sql)
    conn.commit()

    print('TABLE UPDATE COMPLETE')

finally:
    conn.close()