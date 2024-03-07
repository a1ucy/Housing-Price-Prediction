# 使用一个简单的前馈神经网络（Feedforward Neural Network），建立一个模型来预测房屋价格。可以使用包含房屋特征和价格的公开数据集
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.model_selection import train_test_split
import tensorflow as tf
import warnings

warnings.filterwarnings("ignore")
# tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices([], 'GPU')

housing_data = (pd.read_csv('new.csv', encoding='gbk', low_memory=False))
housing_data = housing_data.drop(['url', 'id', 'Cid', 'totalPrice', 'DOM'], axis=1)
housing_data = housing_data[(housing_data['floor'] != '钢混结构') & (housing_data['floor'] != '混合结构')]

floor_type = []


def dataclean_floor(content):
    cn, num = content.split(" ")
    dic = {'未知': 0, '底': 1, '低': 2, '中': 3, '高': 4, '顶': 5}
    floor_type.append(dic[cn])
    return num


def dataclean_time(content):
    return int(content.replace('-', ''))


housing_data['floor'] = housing_data['floor'].apply(dataclean_floor)
housing_data['floorType'] = floor_type
housing_data['tradeTime'] = housing_data['tradeTime'].apply(dataclean_time)
housing_data = housing_data[(housing_data['tradeTime'] >= 20120000) & (housing_data['tradeTime'] < 20180000)]
housing_data = housing_data[housing_data['constructionTime'] != '未知']

df = housing_data.astype('float64').dropna()

# pd.plotting.scatter_matrix(df,figsize=(20,20),alpha=0.1)
# plt.suptitle('Housing Price in Beijing')
# plt.show()
#
# corr = df.corr()
# mask = np.triu(np.ones_like(corr, dtype=bool))
# f, ax = plt.subplots(figsize=(20,20))
# cmap = sns.diverging_palette(230, 20, as_cmap=True)
# sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0, square=True, linewidths=.5, cbar_kws={'shrink': .5}, annot=True, fmt=',.2f')

df_sorted = df.sort_values(by=['tradeTime']).reset_index(drop=True)
print(df_sorted.shape)
X = df_sorted.drop('price', axis=1)
y = df_sorted['price']

x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
X_scaled = x_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1))


def tf_model(X_train_shape1):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(X_train_shape1,)))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))

    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='mse', optimizer=optimizer)
    # model.summary()
    return model


k = 5
patience = 5
batch_size = 64
epochs = 50
callback = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)


def train_model(X, y, bs):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

    model = tf_model(X_train.shape[1])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=bs, verbose=1, validation_split=0.2,
                        callbacks=[callback])
    print('\n')
    test_loss = model.evaluate(X_test, y_test, verbose=1)
    print('\n')

    model_path = f'model_housing.h5'
    model.save(model_path)

    # print(history.history)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


def cross_va(X, y, k, bs, cross_validation):
    scores = []
    cv = TimeSeriesSplit(n_splits=k) if cross_validation == 'time-series' else KFold(n_splits=k, shuffle=True,
                                                                                     random_state=42)
    num = 0
    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = tf_model(X_train.shape[1])
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=bs, verbose=2, validation_split=0.2,
                            callbacks=[callback])
        print('\n')
        test_loss = model.evaluate(X_test, y_test, verbose=1)
        scores.append(round(test_loss, 4))
        num += 1
        print('\n')

        # print(history.history)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(f'Model Loss - {cross_validation} {num}')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()

    print('Scores:', scores)
    print('Scores avg:', np.average(scores), '\n')


train_model(X_scaled, y_scaled, batch_size)

# cross_validation = 'time-series'
# cross_va(X_scaled, y_scaled, k, cross_validation)
# cross_validation = 'k-fold'
# cross_va(X_scaled, y_scaled, k, batch_size, cross_validation)

# test with one of the data from dataset or your own data
# test_data = [[116.386555, 40.0865, 20020601, 0, 89.43, 2, 2, 1, 1, 9, 4, 2009, 1, 6, 0.333, 1, 0, 0, 6, 47574, 3]]
# model_test = load_model('./model_housing.h5')
# test_scaled = x_scaler.fit_transform(test_data)
# prediction = model_test.predict(test_scaled)
# pred = y_scaler.inverse_transform(prediction)
# print(pred)

'''
price = 17053
result = 16746.344
'''
