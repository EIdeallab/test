from keras.layers import Input, Embedding, Dot, Reshape, Dense
from keras.models import Model
from keras.models import Sequential
import UtilStock
import datapreprocess
import pymssql as mssql
import numpy as np
import pandas as pd
import math
import random
random.seed(100)
#################################################
#### Do it here

#data params
train_ratio = 0.7
feature_num = 19
sample_size = 500
len_stock = 0
UNIT = 'WEEK'
SCALER = True
CATEGORICAL = True

#train params
train_epochs = 300

#model save path
MODEL_SAVE_FOLDER_PATH = './model/'
MODEL_NAME = 'STOCK_EMBEDDING'    # choose LSTM /  CONV1D_LSTM / DEEP_CONV1D_LSTM /  CONV2DTD_LSTM

#############################################

def Load_Embedding_Model(stock_dim, link_stock_dim, embedding_size = 10):
    """Model to embed books and wikilinks using the Keras functional API.
       Trained to discern if a link is present in on a book's page"""

    stock = Input(name = 'stock', shape = [1])
    link = Input(name = 'link', shape = [1])

    # 테마 임베딩 (shape will be (None, 1, 50))
    stock_embedding = Embedding(name = 'stock_embedding', input_dim = stock_dim, output_dim = embedding_size)(stock)

    # 링크 임베딩 (shape will be (None, 1, 50))
    link_embedding = Embedding(name = 'link_embedding', input_dim = link_stock_dim, output_dim = embedding_size)(link)

    # 내적으로 테마 임베딩과 링크 임베딩을 한 개의 임베딩 벡터로 변형
    # (shape will be (None, 1, 1))
    merged = Dot(name = 'dot_product', normalize = True, axes = 2)([stock_embedding, link_embedding])

    # 단일 숫자로 shape 변형 (shape will be (None, 1))
    merged = Reshape(target_shape = [1])(merged)

    # 분류를 위한 결과값 출력
    out = Dense(1, activation = 'sigmoid')(merged)
    model = Model(inputs = [stock, link], outputs = out)

    # 원하는 optimizer 와 loss 함수로 모델 학습 시작
    model.compile(optimizer = 'Adam', loss = 'binary_crossentropy',
                  metrics = ['accuracy'])

    return model

def generate_batch(true_pairs, false_pairs, stock_len, link_len, n_positive = 10, negative_ratio = 1.0):

    # batch를 저장할 numpy 배열을 준비합니다.
    batch_size = n_positive * (1 + negative_ratio)
    batch = np.zeros((batch_size, 3))
    true_list = list(true_pairs)
    false_list = list(false_pairs)

    while True:
        # 랜덤으로 True인 샘플을 준비합니다.
        for idx, (stock_id, link_id) in enumerate(random.sample(true_list, n_positive)):
            batch[idx, :] = (stock_id, link_id, 1)

        idx += 1
        # 랜덤으로 False인 샘플을 준비합니다.
        for idx2, (stock_id, link_id) in enumerate(random.sample(false_list, batch_size - n_positive)):
            batch[idx + idx2, :] = (stock_id, link_id, 0)
            
        # 배치에 저장된 데이터들의 순서를 섞습니다.
        np.random.shuffle(batch)
        yield {'stock': batch[:, 0], 'link': batch[:, 1]}, batch[:, 2]


if __name__ == "__main__":
    server, user, password, database = UtilStock.ParseConfig('config.ini')
    connect = mssql.connect(server=server, user=user, password=password, database=database, charset='UTF8')
    cur = connect.cursor()
    info = UtilStock.LoadFinanceStockInfo(cur)

    # LSTM /  CONV1D_LSTM / DEEP_CONV1D_LSTM / CONV2DTD_LSTM
    if MODEL_NAME == 'STOCK_EMBEDDING':
        data, false_data, data_cnt = UtilStock.LoadStockEmbeddingSet(cur)

        stock = np.array(data['BEF_STOCK'])
        link = np.array(data['AFT_STOCK'])
        false_stock = np.array(false_data['BEF_STOCK'])
        false_link = np.array(false_data['AFT_STOCK'])
        stock_dim = np.array(data_cnt['BEF_CNT'])[0]
        link_dim = np.array(data_cnt['AFT_CNT'])[0]
        pairs = np.array((stock,link)).T
        false_pairs = np.array((false_stock,false_link)).T

        model = Load_Embedding_Model(stock_dim, link_dim, 3)
        model.summary()

        n_positive = 4096
        n_negative_ratio = 1
        gen = generate_batch(pairs, false_pairs, stock_dim, link_dim, n_positive, negative_ratio = n_negative_ratio)

        # Train
        h = model.fit_generator(gen, epochs = 1, steps_per_epoch = len(pairs) // (n_positive * (1 + n_negative_ratio)))

        # 임베딩 벡터 추출하기
        stock_layer = model.get_layer('stock_embedding')
        stock_weights = stock_layer.get_weights()[0]
   
        stock_weights = stock_weights / np.linalg.norm(stock_weights, axis = 1).reshape((-1, 1))
        df_s = pd.DataFrame(stock).drop_duplicates().reset_index(drop=True)
        df_w = pd.DataFrame(stock_weights)
        df_em = pd.concat([df_s,df_w], axis=1)
        print(df_em)

        
    else:
        print('Is Not Exist Model')

    #train_model(model, data, label)





