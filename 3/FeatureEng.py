
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import pandas as pd
import numpy as np
import re
import tensorflow as tf
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Flatten, Dense
# from keras.layers.core import Dense
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
from keras.layers.recurrent import LSTM
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge
import os
import h5py
import datetime as dt
import multiprocessing as mp
from statsmodels.api import OLS, add_constant
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import keras
from utility import *
from numpy.random import seed
# seed(10)
import warnings
import time
warnings.filterwarnings("ignore")

config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 8} )
sess = tf.Session(config=config)
keras.backend.set_session(sess)


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        print(f'Elapsed Time - {int(end-start)}')
        return res if res else None
    return wrapper

"""
@timer
func
Equal to
func = timer(func)"""
class Preprocess:
    def preprocess(self, df: pd.DataFrame, threshold: float = 0.0001, time_step=5, num_feature=8, normalize=False) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        label = pd.DataFrame(index=df.index)
        if 'date_str' in df.columns:
            df.drop('date_str', axis=1, inplace=True)

        for idx, name in enumerate(df.columns):
            "Creat training labels"
            if re.match('Y_M_.', name):
                idx = name.split('_')[2]
                label.loc[:, 'Label_{}'.format(idx)] = np.ones([df.shape[0], ])
                label.loc[df[name] < -threshold, 'Label_{}'.format(idx)] = 2
                label.loc[abs(df[name]) <= threshold, 'Label_{}'.format(idx)] = 0

            if int(idx) >= num_feature and normalize:
                "Normalize the dataframe"
                df.loc[:, name] = (df[name] - df[name].mean()) / (df[name].max() - df[name].min())

        "Create a new feature - LOB imbalance"
        df['imbalance'] = df['SP1'].values * df['SV1'].values - df['BV1'].values * df['BP1'].values
        df['imbalance'] = (df['imbalance'] - df['imbalance'].min()) / (df['imbalance'].max() - df['imbalance'].min())


        y_reg = df.iloc[:, 2:7]
        df = df.iloc[:, num_feature:]
        for idx in range(0, 1000 - 1):
            if (df.size - (df.shape[1]) * idx) % (time_step * df.shape[1]) == 0:
                possible_reshape_row = idx
                break

        return df[possible_reshape_row::time_step], label[possible_reshape_row::time_step], y_reg[possible_reshape_row::time_step]

    def pca(self, train_data, test_data, n_components=0.95):
        # scaler = StandardScaler().fit(X=train_data, )
        # train_data = scaler.transform(train_data)
        # test_data = scaler.transform(test_data)
        pca = PCA(n_components, random_state=1)
        pca.fit(train_data)

        print(pca.explained_variance_ratio_)
        self.pca_ratio = np.sum(pca.explained_variance_ratio_)
        return pca.transform(train_data), pca.transform(test_data)
        # return train_data, test_data

class NeuralNet(Preprocess):
    def __init__(self, training, time_step=5, epoch=10, pca_flag=False, batch=256, n_components=4):
        self.training = training
        self.time_step = time_step
        self.epoch = epoch
        self.pca_flag = pca_flag
        self.n_components = n_components
        self.batch = batch

    def create_cls_network(self) -> Sequential:
        seq = Sequential()
        seq.add(LSTM(128, input_shape=(self.time_step, 55), return_sequences=True))
        seq.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
        seq.add(Flatten())
        seq.add(Dense(128, activation='relu'))
        seq.add(Dense(3, activation='softmax'))
        seq.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['categorical_accuracy'])
        return seq

    def create_reg_model(self, ) -> Sequential:
        seq = Sequential()
        seq.add(Dense(units=128, input_dim=self.train_x.shape[1], activation='relu'))
        seq.add(Dense(units=32, activation='relu'))
        seq.add(Dense(units=1))
        seq.compile(loss='mean_squared_error', optimizer='adam')
        return seq

    def run_cls(self):
        self.train_x, self.train_y, _ = self.preprocess(self.training, 0.0015, time_step=self.time_step)
        self.nn = self.create_cls_network()
        self.nn.fit(self.train_x.values.reshape((-1, self.time_step, 55)),
                np_utils.to_categorical(self.train_y['Label_1'], num_classes=3),
                epochs=self.epoch, validation_split=0.05, shuffle=False)

    def predict_cls(self):
        processed_test, y, _ = self.preprocess(self.test, 0.0015, time_step=self.time_step)
        _, test_acc = self.nn.evaluate(x=processed_test.values.reshape(-1, self.time_step, 55), y=np_utils.to_categorical(y['Label_1'], num_classes=3),
                                  steps=self.time_step)
        # print(f'Train acc - {train_acc}')
        print(f'Test acc - {test_acc}')

        y_pred = self.nn.predict(x=processed_test.values.reshape(-1, self.time_step, 55), steps=self.time_step)

    def run_reg(self):
        self.train_x, _, self.train_y = self.preprocess(self.training, 0., time_step=self.time_step, num_feature=14, normalize=False)
        if self.pca_flag:
            self.train_x, self.test_x = self.pca(self.train_x, self.test_x, n_components=self.n_components)

        def min_max_scale(X, range=(0, 1)):
            mi, ma = range
            self.min = X.min()
            self.max = X.max()
            X_std = (X - X.min()) / (X.max() - X.min())
            X_scaled = X_std * (ma - mi) + mi
            return X_scaled


        self.train_x = min_max_scale(self.train_x)


        self.nn = KerasRegressor(build_fn=self.create_reg_model, epochs=self.epoch, batch_size=self.batch,
                validation_split=0.05, shuffle=True, verbose=2)
        self.nn.fit(self.train_x, self.train_y['Y_M_1'])

    def predict_reg(self, test):
        self.test_x, _, self.test_y = self.preprocess(test, 0., time_step=self.time_step, num_feature=14,
                                                      normalize=False)
        self.test_x = (self.test_x - self.min) / (self.max - self.min)
        y_pred = self.nn.predict(self.test_x)
        try:
            r2 = r2_score(self.test_y.Y_M_1, y_pred.reshape(-1,1))
        except:
            r2 = 0
            print('Something wrong')
        print(f'R-square is {r2}')
        # print(f'Mean - y_pred {np.mean(y_pred)}, Mean - y {np.mean(self.test_y.Y_M_1)}')
        return r2


class Regressor(Preprocess):
    def __init__(self, train_data, test_data, num_feature=14, pca_flag=False, n_components=2):
        """
        Regressor for linear_170 regression
        :param df: input data
        :param train_interval: length of training period
        :param test_period: how many days to be tested
        """
        super().__init__()
        self.curr_pointer = 0 # pointed to the index of current date
        self.train_x, _, self.train_y = self.preprocess(df=train_data, time_step=1, num_feature=num_feature,
                                                        normalize=False)
        self.test_x, _, self.test_y = self.preprocess(df=test_data, time_step=1, num_feature=num_feature,
                                                        normalize=False)
        self.pca_flag = pca_flag
        self.n_components = n_components

    def run_regr(self):
        if self.pca_flag == True:
            self.train_x, self.test_x = self.pca(self.train_x, self.test_x, n_components=self.n_components)
        regr = OLS(self.train_y['Y_M_1'], add_constant(self.train_x)).fit()
        # print(regr.summary())
        try:
            y_pred = regr.predict(add_constant(self.test_x))
        except Exception as e:
            print(e)
            return None
        # print(f'R-square is {r2_score(self.test_y.Y_M_1, y_pred)}')
        # print(f'Mean - y_pred {np.mean(y_pred)}, Mean - y {np.mean(self.test_y.Y_M_1)}')
        return r2_score(self.test_y.Y_M_1, y_pred)


    def run_gbrt_regr(self):
        if self.pca_flag == True:
            self.train_x, self.test_x = self.pca(self.train_x, self.test_x, n_components=self.n_components)
        regr = GradientBoostingRegressor(n_estimators=800, max_depth=1, loss = 'ls', learning_rate = .01)
        regr.fit(self.train_x, self.train_y['Y_M_1'])
        # print(regr.summary())
        try:
            y_pred = regr.predict(self.test_x)
        except Exception as e:
            print(e)
            return None
        # print(f'R-square is {r2_score(self.test_y.Y_M_1, y_pred)}')
        # print(f'Mean - y_pred {np.mean(y_pred)}, Mean - y {np.mean(self.test_y.Y_M_1)}')
        return r2_score(self.test_y.Y_M_1, y_pred)

    def run_lasso_regr(self):
        if self.pca_flag == True:
            self.train_x, self.test_x = self.pca(self.train_x, self.test_x, n_components=self.n_components)
        regr = Ridge()
        regr.fit(self.train_x, self.train_y['Y_M_1'])
        # print(regr.summary())
        try:
            y_pred = regr.predict(self.test_x)
        except Exception as e:
            print(e)
            return None
        # print(f'R-square is {r2_score(self.test_y.Y_M_1, y_pred)}')
        # print(f'Mean - y_pred {np.mean(y_pred)}, Mean - y {np.mean(self.test_y.Y_M_1)}')
        return r2_score(self.test_y.Y_M_1, y_pred)
if __name__ == '__main__':
    # comp_BM(file='data/step1_1101.csv')
    # concat_data()

    """
    Splite the data
    """
    # split_dataset()

    """
    Preprocess dataframe and reshape the data fed into neural networks
    """
    # files = os.listdir('data/')
    # files = [file for file in files if os.path.isfile('data/'+file)]
    # files = [(file, int(file.split('.')[0].split('_')[1])) for file in files if file.split('.')[1] == 'csv' ]
    # sort_files = sorted(files, key=lambda x:x[1])
    # sort_files = [file for file in sort_files if file[1] != 2412]
    #
    # for file, ticker in sort_files:
    #     print(f'Doing - {ticker}')
    #     df = pd.read_csv('data/'+file, index_col=0)
    #     df['date_label'] = df.groupby(['date']).ngroup()
    #     n_components = 1
    #
    #     train_end = 170
    #     train_start = 0
    #
    #     train_data = df.loc[(df['date_label'] <= train_end) & (df['date_label'] >= train_start)]
    #     train_data.drop('date_label', axis=1, inplace=True)
    #
    #     epoch = 20
    #     batch = 64
    #     pca_flag = False
    #
    #     nn = NeuralNet(training=train_data, time_step=2, epoch=epoch, batch=batch, pca_flag=pca_flag,
    #                    n_components=.95)
    #     nn.run_reg()
    #     record = []
    #     for test_end in range(171, df['date_label'][-1] + 1):
    #
    #         test_data = df.loc[(df['date_label'] <= test_end) & (df['date_label'] > test_end - 1)]
    #         test_data.drop('date_label', axis=1, inplace=True)
    #
    #         # try:
    #         #     if os.path.exists(f'result/nn/nn_reg_{ticker}.csv'):
    #         #         continue
    #         # except:
    #         #     pass
    #
    #         r2 = nn.predict_reg(test_data)
    #         record.append([test_data.date[0], r2])
    #     record = pd.DataFrame(record, columns=['date', 'r2'])
    #     record.index = pd.to_datetime(record.date, format='%Y-%m-%d')
    #     record.sort_index(inplace=True)
    #     # record.to_csv(f'result/linear_20//linear_reg_pca-{n_components}_without_norm.csv', index=False)
    #     try:
    #         record.to_csv(f'result/nn/nn_reg_{ticker}.csv', index=False)
    #     except:
    #         os.mkdir(f'result/nn/nn_no_pca_{ticker}/')
    #         record.to_csv(f'result/nn/nn_reg_{ticker}.csv', index=False)

        # nn = NeuralNet(training=train, test=test, time_step=30, epoch=1)
        # nn.run_cls()
        # nn.predict_cls()


    """
    1.Preprocess dataframe regarding to different time scale
    2.Build the regression model
    """
    # train_path = 'data/training/0050_28Train/'
    # test_path = 'data/test/0050_28Test/'
    # files = os.listdir(train_path)
    #
    # for n_components in range(1,40,5):
    #     record = []
    #     print(f'Current number of component - {n_components}')
    #     for train_file in files:
    #         if train_file[-3:] == 'csv':
    #             try:
    #                 train = pd.read_csv(train_path+train_file, index_col=0)
    #                 idx = train_file.split('_')[1].split('.')[0]
    #                 test = pd.read_csv(test_path+f'test_{idx}.csv', index_col=0)
    #                 reg = Regressor(train_data=train, test_data=test, pca_flag=True, n_components=n_components)
    #                 r2 = reg.run_regr()
    #                 record.append([test.date[0], r2, reg.pca_ratio])
    #             except:
    #                 continue
    #     record = pd.DataFrame(record, columns=['date', 'r2',  'pca_ratio'])
    #     record.index = pd.to_datetime(record.date, format='%Y-%m-%d')
    #     record.sort_index(inplace=True)
    #     record.to_csv(f'result/linear_20//linear_reg_pca-{n_components}_without_norm.csv', index=False)
        # record.to_csv(f'result/linear_20/linear_reg.csv', index=False)

    # TRAIN_DURATION = 170
    """
    1.Preprocess dataframe regarding to different time scale
    2.Build the regression model for top 50 stocks
    """
    files = os.listdir('data/')
    files = [file for file in files if os.path.isfile('data/'+file)]
    files = [(file, int(file.split('.')[0].split('_')[1])) for file in files if file.split('.')[1] == 'csv' ]
    sort_files = sorted(files, key=lambda x:x[1])
    sort_files = [file for file in sort_files[1:] if file[1] != 2412]

    for file, ticker in sort_files:
        print(f'Doing - {ticker}')
        df = pd.read_csv('data/'+file, index_col=0)
        df['date_label'] = df.groupby(['date']).ngroup()
        n_components = 1

        def mp_reg(epoch):
            print(f'Epoch - {epoch}')
            train_start = 170 + epoch - TRAIN_DURATION # 0
            train_end = 170 + epoch  - 1  # 179

            test_end = 170 + epoch  # 180

            train_data = df.loc[(df['date_label'] <= train_end) & (df['date_label'] >= train_start)]
            train_data.drop('date_label', axis=1, inplace=True)

            test_data = df.loc[(df['date_label'] <= test_end) & (df['date_label'] > train_end)]
            test_data.drop('date_label', axis=1, inplace=True)

            reg = Regressor(train_data=train_data, test_data=test_data, pca_flag=False, n_components=n_components)
            r2 = reg.run_regr()
            return test_data.date[0], r2

        for idx in range(5, 171, 5):
            if os.path.exists(f'result/linear/linear_no_pca_{ticker}/linear_reg_{idx}.csv'):
                continue

            TRAIN_DURATION = idx

            pool = mp.Pool(mp.cpu_count())
            record = pool.map(mp_reg, range(0, df.date_label[-1]-170))
            pool.close()

            record = pd.DataFrame(record, columns=['date', 'r2'])
            record.index = pd.to_datetime(record.date, format='%Y-%m-%d')
            record.sort_index(inplace=True)
            # record.to_csv(f'result/linear_20//linear_reg_pca-{n_components}_without_norm.csv', index=False)
            try:
                record.to_csv(f'result/linear/linear_no_pca_{ticker}/linear_reg_{TRAIN_DURATION}.csv', index=False)
            except:
                os.mkdir(f'result/linear/linear_no_pca_{ticker}/')
                record.to_csv(f'result/linear/linear_no_pca_{ticker}/linear_reg_{TRAIN_DURATION}.csv', index=False)


    """
    1.Preprocess dataframe regarding to different time scale
    2.Build the GBRT regression model for top 50 stocks
    """
    files = os.listdir('data/')
    files = [file for file in files if os.path.isfile('data/'+file)]
    files = [(file, int(file.split('.')[0].split('_')[1])) for file in files if file.split('.')[1] == 'csv' ]
    sort_files = sorted(files, key=lambda x:x[1])
    sort_files = [('step1_0050.csv', '0050')] + [file for file in sort_files[1:] if file[1] != 2412]

    for file, ticker in sort_files:
        print(f'Doing - {ticker}')
        df = pd.read_csv('data/'+file, index_col=0)
        df['date_label'] = df.groupby(['date']).ngroup()
        n_components = 1

        def mp_reg(epoch):
            print(f'Epoch - {epoch}')
            train_start = 170 + epoch - TRAIN_DURATION # 0
            train_end = 170 + epoch  - 1  # 179

            test_end = 170 + epoch  # 180

            train_data = df.loc[(df['date_label'] <= train_end) & (df['date_label'] >= train_start)]
            train_data.drop('date_label', axis=1, inplace=True)

            test_data = df.loc[(df['date_label'] <= test_end) & (df['date_label'] > train_end)]
            test_data.drop('date_label', axis=1, inplace=True)

            reg = Regressor(train_data=train_data, test_data=test_data, pca_flag=False, n_components=n_components)
            r2 = reg.run_gbrt_regr()
            return test_data.date[0], r2

        for idx in [20, 10]:
            if os.path.exists(f'result/gbrt/gbrt_no_pca_{ticker}/gbrt_reg_{idx}.csv'):
                continue

            TRAIN_DURATION = idx

            pool = mp.Pool(mp.cpu_count())
            record = pool.map(mp_reg, range(0, df.date_label[-1]-170))
            pool.close()

            record = pd.DataFrame(record, columns=['date', 'r2'])
            record.index = pd.to_datetime(record.date, format='%Y-%m-%d')
            record.sort_index(inplace=True)
            # record.to_csv(f'result/linear_20//linear_reg_pca-{n_components}_without_norm.csv', index=False)
            try:
                record.to_csv(f'result/gbrt/gbrt_no_pca_{ticker}/gbrt_reg_{TRAIN_DURATION}.csv', index=False)
            except:
                os.mkdir(f'result/gbrt/gbrt_no_pca_{ticker}/')
                record.to_csv(f'result/gbrt/gbrt_no_pca_{ticker}/gbrt_reg_{TRAIN_DURATION}.csv', index=False)

    """
    1.Preprocess dataframe regarding to different time scale
    2.Build the Lasso regression model for top 50 stocks
    """
    files = os.listdir('data/')
    files = [file for file in files if os.path.isfile('data/'+file)]
    files = [(file, int(file.split('.')[0].split('_')[1])) for file in files if file.split('.')[1] == 'csv' ]
    sort_files = sorted(files, key=lambda x:x[1])
    sort_files = [('step1_0050.csv', '0050')] + [file for file in sort_files[1:] if file[1] != 2412]

    for file, ticker in sort_files:
        print(f'Doing - {ticker}')
        df = pd.read_csv('data/'+file, index_col=0)
        df['date_label'] = df.groupby(['date']).ngroup()
        n_components = 1

        def mp_reg(epoch):
            print(f'Epoch - {epoch}')
            train_start = 170 + epoch - TRAIN_DURATION # 0
            train_end = 170 + epoch  - 1  # 179

            test_end = 170 + epoch  # 180

            train_data = df.loc[(df['date_label'] <= train_end) & (df['date_label'] >= train_start)]
            train_data.drop('date_label', axis=1, inplace=True)

            test_data = df.loc[(df['date_label'] <= test_end) & (df['date_label'] > train_end)]
            test_data.drop('date_label', axis=1, inplace=True)

            reg = Regressor(train_data=train_data, test_data=test_data, pca_flag=False, n_components=n_components)
            r2 = reg.run_lasso_regr()
            return test_data.date[0], r2

        for idx in range(5, 171, 15):
            if os.path.exists(f'result/ridge/ridge_no_pca_{ticker}/ridge_reg_{idx}.csv'):
                continue

            TRAIN_DURATION = idx

            pool = mp.Pool(mp.cpu_count())
            record = pool.map(mp_reg, range(0, df.date_label[-1]-170))
            pool.close()

            record = pd.DataFrame(record, columns=['date', 'r2'])
            record.index = pd.to_datetime(record.date, format='%Y-%m-%d')
            record.sort_index(inplace=True)
            # record.to_csv(f'result/linear_20//linear_reg_pca-{n_components}_without_norm.csv', index=False)
            try:
                record.to_csv(f'result/ridge/ridge_no_pca_{ticker}/ridge_reg_{TRAIN_DURATION}.csv', index=False)
            except:
                os.mkdir(f'result/ridge/ridge_no_pca_{ticker}/')
                record.to_csv(f'result/ridge/ridge_no_pca_{ticker}/ridge_reg_{TRAIN_DURATION}.csv', index=False)


