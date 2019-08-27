import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.metrics import r2_score
import os
import multiprocessing as mp
import warnings
import time
warnings.filterwarnings("ignore")
def create_feature(df: pd.DataFrame):
    corr_mat = df.corr()
    plot_corr(df, corr_mat)

    groups = {}
    for group in df.columns:
        if group.split('_')[0] not in groups:
            groups[group.split('_')[0]] = group

    output = pd.DataFrame()
    for col1, col2 in combinations(groups.values(), 2):
        if corr_mat.loc[col1, col2] > 0:
            output[col1 + '*' + col2] = df[col1] * df[col2]
    return output

def plot_corr(df, corr_mat, filename='original_features_corr.png'):
    f = plt.figure(figsize=(16, 10))
    plt.matshow(corr_mat, fignum=f.number)
    plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=45)
    plt.yticks(range(df.shape[1]), df.columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16)
    plt.show()
    # plt.savefig(fname=filename, dpi=900)

def run_regr(train, test):

    regr = sm.OLS(train['Y_M_1'], sm.add_constant(train.iloc[:, 14:-1])).fit()
    # print(regr.summary())
    try:
        y_pred = regr.predict(sm.add_constant(test.iloc[:, 14:-1]))
    except Exception as e:
        print(e)
        return None
    # print(f'R-square is {r2_score(self.test_y.Y_M_1, y_pred)}')
    # print(f'Mean - y_pred {np.mean(y_pred)}, Mean - y {np.mean(self.test_y.Y_M_1)}')
    return r2_score(test.Y_M_1, y_pred), regr.tvalues

if __name__ == '__main__':
    df = pd.read_csv('data/step1_0050.csv', index_col=0)
    ticker = '0050'
    added_features = create_feature(df.iloc[:,14:])
    df = pd.concat([df, added_features], axis=1).dropna()
    df['date_label'] = df.groupby(['date']).ngroup()

    def mp_reg(epoch):
        print(f'Epoch - {epoch}')
        train_start = 170 + epoch - TRAIN_DURATION # 0
        train_end = 170 + epoch  - 1  # 179

        test_end = 170 + epoch  # 180

        train_data = df[(df['date_label'] <= train_end) & (df['date_label'] >= train_start)]
        train_data.drop('date_label', axis=1, inplace=True)

        test_data = df[(df['date_label'] <= test_end) & (df['date_label'] > train_end)]
        test_data.drop('date_label', axis=1, inplace=True)

        r2, tvalues = run_regr(train=train_data, test=test_data)

        try:
            tvalues.to_csv(f'result/Tvalues/{ticker}/Tvalue_reg_{test_data.date[0]}.csv', index=True)
        except:
            os.mkdir(f'result/Tvalues/{ticker}')
            tvalues.to_csv(f'result/Tvalues/{ticker}/Tvalue_reg_{test_data.date[0]}.csv', index=True)

        return test_data.date[0], r2,

    # for idx in range(5, 171, 5):
    for idx in [170]:
        # if os.path.exists(f'result/Tvalues/{ticker}/linear_reg_{idx}.csv'):
        #     continue

        TRAIN_DURATION = idx

        pool = mp.Pool(mp.cpu_count())
        record = pool.map(mp_reg, range(0, df.date_label[-1]-170))
        pool.close()

        record = pd.DataFrame(record, columns=['date', 'r2'])
        record.index = pd.to_datetime(record.date, format='%Y-%m-%d')
        record.sort_index(inplace=True)

        try:
            record.to_csv(f'result/Tvalues/{ticker}/linear_reg_{TRAIN_DURATION}.csv', index=False)
        except:
            os.mkdir(f'result/Tvalues/{ticker}')
            record.to_csv(f'result/Tvalues/{ticker}/linear_reg_{TRAIN_DURATION}.csv', index=False)


