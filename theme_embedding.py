from keras.layers import Input, Embedding, Dot, Reshape, Dense
from keras.models import Model
from keras.models import Sequential
import UtilStock
import datapreprocess
import pymssql as mssql
import numpy as np
import math
import random
random.seed(100)
#################################################
#### Do it here

#data params
train_ratio = 0.7
feature_num = 19
sample_size = 500
len_theme = 0
UNIT = 'WEEK'
SCALER = True
CATEGORICAL = True

#train params
train_epochs = 300

#model save path
MODEL_SAVE_FOLDER_PATH = './model/'
MODEL_NAME = 'THEME_EMBEDDING'    # choose LSTM /  CONV1D_LSTM / DEEP_CONV1D_LSTM /  CONV2DTD_LSTM

#############################################

def Load_Embedding_Model(theme, link_theme, theme_dim, link_theme_dim, embedding_size = 10, classification = False):
    """Model to embed books and wikilinks using the Keras functional API.
       Trained to discern if a link is present in on a book's page"""

    # 테마 임베딩 (shape will be (None, 1, 50))
    theme_embedding = Embedding(name = 'bef_embedding', input_dim = theme_dim, output_dim = embedding_size)(theme)

    # 링크 임베딩 (shape will be (None, 1, 50))
    link_embedding = Embedding(name = 'aft_embedding', input_dim = link_theme_dim, output_dim = embedding_size)(link_theme)

    # 내적으로 테마 임베딩과 링크 임베딩을 한 개의 임베딩 벡터로 변형
    # (shape will be (None, 1, 1))
    merged = Dot(name = 'dot_product', normalize = True, axes = 2)([theme_embedding, link_embedding])

    # 단일 숫자로 shape 변형 (shape will be (None, 1))
    merged = Reshape(target_shape = [1])(merged)

    # 분류를 위한 결과값 출력
    out = Dense(1, activation = 'sigmoid')(merged)
    model = Model(inputs = [theme, link_theme], outputs = out)

    # 원하는 optimizer 와 loss 함수로 모델 학습 시작
    model.compile(optimizer = 'Adam', loss = 'binary_crossentropy',
                  metrics = ['accuracy'])

    return model

def generate_batch(theme_len, link_len, true_pairs, n_positive = 10, negative_ratio = 1.0):

    # batch를 저장할 numpy 배열을 준비합니다.
    batch_size = n_positive * (1 + negative_ratio)
    batch = np.zeros((batch_size, 3))

    while True:
        # 랜덤으로 True인 샘플을 준비합니다.
        for idx, (theme_id, link_id) in enumerate(random.sample(true_pairs, n_positive)):
            batch[idx, :] = (theme_id, link_id, 1)
        idx += 1

        # 배치 사이즈가 다 찰 때까지 False인 샘플을 추가합니다.
        while idx < batch_size:

            # Random selection
            random_theme = random.randrange(theme_len)
            random_link = random.randrange(link_len)

            # True인 샘플이 아니라는 것(False인 샘플이라는 것)을 체크합니다.
            if (random_theme, random_link) not in true_pairs:

                # False인 샘플을 배치에 추가합니다.
                batch[idx, :] = (random_theme, random_link, 0)
                idx += 1

        # 배치에 저장된 데이터들의 순서를 섞습니다.
        np.random.shuffle(batch)
        yield {'theme': batch[:, 0], 'link': batch[:, 1]}, batch[:, 2]

if __name__ == "__main__":
    server, user, password, database = UtilStock.ParseConfig('config.ini')
    connect = mssql.connect(server=server, user=user, password=password, database=database, charset='UTF8')
    cur = connect.cursor()
    info = UtilStock.LoadFinanceStockInfo(cur)

    # LSTM /  CONV1D_LSTM / DEEP_CONV1D_LSTM / CONV2DTD_LSTM
    if MODEL_NAME == 'THEME_EMBEDDING':
        data, data_cnt = UtilStock.LoadThemeEmbeddingSet(cur)
        
        model = Load_Embedding_Model(data, data_cnt)
    else:
        print('Is Not Exist Model')

    if CATEGORICAL:
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
    else:
        model.compile(loss='mse', optimizer='rmsprop', metrics=['acc'])
    print('Model Build...')
    model.summary()

    train_model(model, data, label)





