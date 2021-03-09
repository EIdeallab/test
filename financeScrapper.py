import pandas as pd
import pymssql as mssql
import time
import configparser
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from re import sub
from decimal import Decimal
import UtilStock

def financeScrap():
    # mssql 서버 접속
    sv, us, ps, db = UtilStock.ParseConfig('config.ini')

    conn = mssql.connect(server=sv, user=us, password=ps,database=db, charset= 'UTF8')
    try:
        cur = conn.cursor()

        # 주식 재무정보 테이블 생성
        sql = "EXEC CREATE_TBL_FINANCE_INFO"
        cur.execute(sql)
        conn.commit()
        
        # Sql 서버에서 주식 종목 리스트 로드

        sql = "SELECT STOCK_CODE FROM STOCK_INFO;"
        print("sql:", sql)
        cur.execute(sql)
        name_df =  pd.DataFrame(cur.fetchall())

        # 네이버 주식 데이터 가져오기
        data = [item[0] for item in name_df.values]

        # 웹 크롤러 생성
        driver = webdriver.Chrome()

        for stockCode in data:
            stockCode = stockCode.strip()
            
            url = 'https://navercomp.wisereport.co.kr/v2/company/c1010001.aspx?cmp_cd={code}&amp;target=finsum_more'.format(code=stockCode)
            print("Load Stock financial info : ", url)

            # 재무제표 페이지 가져오기
            driver.get(url)
            try : driver.find_element_by_xpath("//a[@id='cns_Tab22']").click()
            except : continue

            html = driver.page_source
            soup = BeautifulSoup(html, 'lxml')

            # 주요재무정보 테이블 가져오기(분기별)
            fn_cells = soup.select('table.gHead01.all-width')

            # 데이터프레임 가공
            df = UtilStock.ProcessFinance(fn_cells)

            
            for name, val in df.iteritems():
            # 얻은 데이터가 DB에 없는 경우 업로드
                sql = "DECLARE @rtn int "
                sql += "EXEC @rtn = UPDATE_FINANCE_INFO \
                {0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}, \
                {11}, {12},{13}, {14}, {15}, '{16}' \
                SELECT @rtn".format(
                    stockCode, 
                    val['매출액'], 
                    val['영업이익'], 
                    val['당기순이익'], 
                    val['자산총계'], 
                    val['부채총계'],
                    val['자본총계'],
                    val['영업활동현금흐름'], 
                    val['투자활동현금흐름'], 
                    val['재무활동현금흐름'], 
                    val['CAPEX'], 
                    val['FCF'], 
                    val['이자발생부채'], 
                    val['자본유보율'], 
                    val['현금배당수익률'], 
                    val['발행주식수(보통주)'], 
                    name
                )
                cur.execute(sql)
                # 요청 후 대기(트래픽 과다 방지)
                conn.commit()

            # 요청 후 대기(트래픽 과다 방지)
            time.sleep(0.05)
            
    finally:
        conn.close()

