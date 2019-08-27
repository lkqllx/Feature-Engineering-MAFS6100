import pandas as pd
import numpy as np
import re
import os
import h5py
import datetime as dt

TRAIN_DURATION = 28


def parse_date(date):
    return dt.datetime.strptime(date, '%Y-%m-%d')

def next_weekday(date):
    date += dt.timedelta(days=1)
    for _ in range(7):
        if date.weekday() < 5:
            break
        else:
            date += dt.timedelta(days=1)
    return date

def concat_data(path='data/'):
    files = os.listdir(path)
    files = [file for file in files if file != '.DS_Store']
    save_to = 'data/all_data.h5'
    for idx, file in enumerate(files):
        curr_df = pd.read_csv(path+file)
        if idx == 0:
            curr_df.to_hdf(save_to, 'data', mode='w', format='table')
            del curr_df
        else:
            curr_df.to_hdf(save_to, 'data', append=True)
            del curr_df
        if idx % 5 == 0:
            print(f'Processing {idx}')

def split_dataset(path = 'data/'):
    files = os.listdir(path)
    files = [file for file in files if file[-3:] == 'csv']
    test_file = 'step1_0050.csv'
    ticker = test_file.split('.')[0].split('_')[1]
    if not os.path.exists(f'data/training/{ticker}_{TRAIN_DURATION}Train'):
        os.mkdir(f'data/training/{ticker}_{TRAIN_DURATION}Train')
    if not os.path.exists(f'data/test/{ticker}_{TRAIN_DURATION}Test'):
        os.mkdir(f'data/test/{ticker}_{TRAIN_DURATION}Test')
    test_df = pd.read_csv('data/'+test_file, index_col=0).dropna()
    test_df['date_str'] = test_df['date'].values
    test_df.date = test_df['date'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d'))

    date_set = set(test_df['date_str'].values.tolist())
    date_key = {date: dt.datetime.strptime(date, '%Y-%m-%d') for date in date_set}
    date_key = sorted(date_key.items(), key=lambda x:x[1])

    """Decide how many training and testing sample"""
    global sorted_date_set
    sorted_date_set = [date for date, _ in date_key][3:33]



    """
    Initialize directory and test file
    """
    for idx, curr_date in enumerate(sorted_date_set):
        if idx % 1 == 0:
            print(f'Test Done - {idx}')

        training = fetch_training(test_df, training_start=curr_date, duration=TRAIN_DURATION)
        test_date = dt.datetime.strptime(curr_date, '%Y-%m-%d') + dt.timedelta(days=TRAIN_DURATION)
        for _ in range(100):
            if test_date.strftime('%Y-%m-%d') in date_set:
                break
            else:
                test_date += dt.timedelta(days=1)
        testing = fetch_testing(test_df, testing_start=test_date.strftime('%Y-%m-%d'), duration=1, curr_date_set=date_set)
        training.to_csv(f'data/training/{ticker}_{TRAIN_DURATION}Train/training_{idx}.csv')
        testing.to_csv(f'data/test/{ticker}_{TRAIN_DURATION}Test/test_{idx}.csv')


    """Append training data to different test period"""
    # for count, file in enumerate(files[11:]):
    #     count += 11
    #     print('-'*20, f'{file} - Count {count}', '-'*20)
    #     try:
    #         if file.split('.')[1] == 'csv' and file != test_file:
    #             df = pd.read_csv(path+file, index_col=0)
    #             df.date = df['date'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d'))
    #             for idx, curr_date in enumerate(sorted_date_set):
    #                 if idx % 10 == 0:
    #                     print(f'{file} Done - {idx}')
    #                 training = fetch_training(df, training_start=curr_date, duration=TRAIN_DURATION)
    #                 prev_training = pd.read_csv(f'data/training/training_{idx}.csv', index_col=0)
    #                 prev_training = prev_training.append(training, sort=False)
    #                 prev_training.to_csv(f'data/training/{ticker}/training_{idx}.csv')
    #     except Exception as e:
    #         print(f'{e} in {file}')
    #         continue

    """multi-processing the concat function"""
    # print(mp.cpu_count())
    # pool = mp.Pool(mp.cpu_count())
    # pool.map(concat_training, files[24:])

def concat_training(file):
    path = 'data/'
    test_file = 'step1_0050.csv'

    # for count, file in enumerate(files[11:]):
    #     count += 11
    print('-'*20, f'{file}', '-'*20)
    try:
        if file.split('.')[1] == 'csv' and file != test_file:
            df = pd.read_csv(path+file, index_col=0)
            df.date = df['date'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d'))
            for idx, curr_date in enumerate(sorted_date_set):
                if idx % 10 == 0:
                    print(f'{file} Done - {idx}')
                training = fetch_training(df, training_start=curr_date, duration=TRAIN_DURATION)
                prev_training = pd.read_csv(f'data/training/training_{idx}.csv', index_col=0)
                prev_training = prev_training.append(training, sort=False)
                prev_training.to_csv(f'data/training/training_{idx}.csv')
    except Exception as e:
        print(f'{e} in {file}')



def fetch_training(df, training_start, duration):
    """
    Fetch the training data for a fixed period
    :return: output - in that duration
    """
    # if not dt.datetime.strptime(training_start, '%Y-%m-%d').weekday() < 5:
    #     print('Start date should be weekday!')
    #     return False

    dt_training_start = dt.datetime.strptime(training_start, '%Y-%m-%d')
    dt_training_end = dt_training_start + dt.timedelta(days=duration)
    try:
        return df.loc[(df.date < dt_training_end) & (df.date >= dt_training_start)]

    except:
        print('Error fetch_training')
        return False

def fetch_testing(df, testing_start, curr_date_set, duration=1):
    """
    Fetch the testing data for a fixed period but the testing start may not in the dataframe
    So we need to look for the next available start date
    :return: output - in that duration
    """
    dt_testing_start = dt.datetime.strptime(testing_start, '%Y-%m-%d')
    for _ in range(200):
        if testing_start in curr_date_set:
            dt_testing_end = dt_testing_start + dt.timedelta(days=duration)
            break
        else:
            dt_testing_start = next_weekday(dt_testing_start)
    try:
        return df.loc[(df.date < dt_testing_end) & (df.date >= dt_testing_start)]

    except:
        print('Error fectch_testing')
        return False