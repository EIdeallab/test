import pandas as pd
import pymssql as mssql
import os
import sys
import time
import configparser
import requests
import urllib.request
import json
from io import StringIO
from selenium import webdriver
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import UtilStock

def themeScrap():
    # 기본 정보 로드
    sv, us, ps, db = UtilStock.ParseConfig('config.ini')

    # mssql 서버 접속
    conn = mssql.connect(server=sv, user=us, password=ps,database=db, charset= 'UTF8')
    try:
        cur = conn.cursor()

        # 주식 테마정보 테이블 생성
        sql = "EXEC CREATE_TBL_THEME_INFO"
        cur.execute(sql)
        conn.commit()

        # 주식 이름정보 가져와서 미리 저장
        sql = "SELECT STOCK_CODE, STOCK_NAME FROM STOCK_INFO"
        print("sql:", sql)
        cur.execute(sql)
        name_df = pd.DataFrame(cur.fetchall())
        name_df.index = name_df[1]
        conn.commit()

        driver = webdriver.Chrome()
        url = "http://www.paxnet.co.kr/stock/infoStock/thema"
        driver.get(url)

        print("Load Thema info : ", url)
        # 총 5 페이지의 테마 섹터에 대하여 조사 (Default 1 Page)
        theme_code = 0
        for page in range(2, 5): 
            # 앞의 30개는 유효하지 않으므로 뒤의 30개에 대해서 선택
            for index in range(30, 60):
                xpath = "//td[@class='ellipsis']//a[starts-with(@href, 'javascript')]"
                elements = driver.find_elements_by_xpath(xpath)
                theme_name = elements[index].text
                elements[index].click()

                # 데이터 프레임 가져오기
                html = driver.page_source
                soup = BeautifulSoup(html)
                descript = soup.select("div[class='descript']")
                table = soup.select("table[class='table-data']")

                descript = descript[0].text
                table_df = pd.read_html(str(table[0]))[0]

                # 테마 코드별 정보 생성
                sql = "UPDATE_THEME_INFO {0}, '{1}', '{2}'".format(theme_code, theme_name, descript)
                cur.execute(sql)
                conn.commit()

                # 각 주식별 테마 코드 입력
                for stock_name in table_df.iterrows(): 
                    try : 
                        stock_code = name_df.loc[stock_name[1][0]][0]
                    except : 
                        print(stock_name[1][0])
                        continue
                    sql = "UPDATE_STOCK_THEME {0}, {1}".format(stock_code, theme_code)
                    cur.execute(sql)
                
                conn.commit()
                theme_code += 1
                driver.back()

            # 다음 페이지 버튼 클릭
            xpath = "//a[@href='#' and text()='{0}']".format(page)
            driver.find_element_by_xpath(xpath).click()
            
            
        # 주식 재무제표 데이터 일자 가져오기
        sql = "SELECT TOP 1 A.[DATE] FROM FINANCE_INFO A ORDER BY [DATE] DESC"
        cur.execute(sql)
        date_t =  cur.fetchall()
        date = datetime.strptime(date_t[0][0], '%Y-%m-%d').date()
    finally:
        conn.close()



