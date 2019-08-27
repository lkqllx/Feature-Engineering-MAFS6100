 # -*- coding: utf-8 -*-
"""
This module can:
    1. draw the heatmap of oosR2 for given ticker, model and label
    2. draw the oosR2 line chart for given ticker, model and label
To use it, 
    1. uncomment/comment the parts you need/needn't in the last part.
    2. modify the corresponding path, ticker, model and label.
    3. run this module.
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
import seaborn as sns


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def plot_oosR2(path_df='C://Users//EnhaoDaniel//111mafs6100C independent project feature engineering//programs//Scikit_results//modelStats_multiProcesses_0050_all_past.csv',
               var=1,
               value='gbrt_oosR2'):

    df = pd.read_csv(path_df, header=0, sep=',')
    df_list = np.split(df, df[df.iloc[:, 1:].isnull().all(1)].index)
    
    numOfPastDays_list=[]
#    numOfForwardDays=[]
    value_list=[]
    
    for idx in range(1,len(df_list)):
    #    df_list[idx].reset_index(drop=True, inplace=True)
        numOfPastDays_list.append(float(df_list[idx].iloc[0, 0]))
        value_list.append(df_list[idx][value].iloc[var])
    
    plt.plot(numOfPastDays_list, value_list)
    plt.legend([str(value) + ' ' + str(var)])
    plt.show()
    
def heatmap_fun(path_input='C://Users//EnhaoDaniel//111mafs6100C independent project feature engineering//programs//Scikit_results//0050_numOfPastDays=[125,130,135,140,165,170,175,180]//',
               idx_of_label=3,  # choose from [0,1,2,3,4]
               model='gbrt_oosR2'):
    os.chdir(path_input)
    input_file_list = [pd.read_csv(file) for file in os.listdir(path_input)]
    
#    heatmap_df = pd.DataFrame(columns=heatmap_column)
    heatmap_column_list = []
    for heatmap_column_idx, df in enumerate(input_file_list):
        training_start_date_length = len(np.unique(df['training_start_date']))
        curr_column_list = []
        for num_of_testday in range(0,20,1):
            start = idx_of_label + num_of_testday*5
            end = start + training_start_date_length*100
            curr_column_list.append(df[model].iloc[start:end:100].mean())
        heatmap_column_list.append(pd.Series(curr_column_list))
    heatmap_df = pd.concat(heatmap_column_list, axis=1, ignore_index=True)
    heatmap_df.columns = ['125','130','135','140','165','170','175','180']
    
    return heatmap_df

        


if __name__ == "__main__":
#    """the oosR2 heatmap for (2 stocks) * (5 models) * (5 labels)"""
#    for ticker in ['0050', '2330']:
#        path_input='C://Users//EnhaoDaniel//111mafs6100C independent project feature engineering//programs//Scikit_results//' + str(ticker) + '_numOfPastDays=[125,130,135,140,165,170,175,180]//'
#        for model in ['gbrt_oosR2', 'linear_oosR2', 'plsr_oosR2', 'lasso_oosR2', 'nn_oosR2']:
#            for idx_of_label in range(5):
##                print('current model is ' + str(model) + '\n' + 'current index of label is ' + str(idx_of_label))
#                heatmap_df = heatmap_fun(path_input=path_input,idx_of_label=idx_of_label,model=model)
#                plt.figure(figsize=(20,7))
#                sns.heatmap(heatmap_df)
#                plt.title('oosR2 for ' + 'stock ' + str(ticker) + ', ' + str(model) + ', Y_M_' + str(idx_of_label + 1))
#                plt.xlabel('numOfdays in training set')
#                plt.ylabel('number of prediction day')
#                plt.show()
    
    """the average oosR2 heatmap for (2 stocks) * (5 models)"""
    for ticker in ['0050', '2330']:
        path_input='C://Users//EnhaoDaniel//111mafs6100C independent project feature engineering//programs//Scikit_results//' + str(ticker) + '_numOfPastDays=[125,130,135,140,165,170,175,180]//'
        for model in ['gbrt_oosR2', 'linear_oosR2', 'plsr_oosR2', 'lasso_oosR2', 'nn_oosR2']:
            idx_of_label = 0
            heatmap_df = heatmap_fun(path_input=path_input, idx_of_label=idx_of_label, model=model)
            for idx_of_label in range(1,5,1):
                heatmap_df = heatmap_df + heatmap_fun(path_input=path_input, idx_of_label=idx_of_label,model=model)
                heatmap_df /= 5
            plt.figure(figsize=(20,8))
            sns.heatmap(heatmap_df)
            plt.title('average oosR2 for ' + str(model) + ' of stock ' + str(ticker))
            plt.xlabel('numOfdays in training set')
            plt.ylabel('number of prediction day')
            plt.show()
    
#    for ticker in ['0050']:
#        path_df='C://Users//EnhaoDaniel//111mafs6100C independent project feature engineering//programs//Scikit_results//modelStats_multiProcesses_' + str(ticker) + '_all_past.csv'
#        for value in ['gbrt_oosR2','linear_oosR2','plsr_oosR2','lasso_oosR2','nn_oosR2']:
#            for var in range(1,6,1):
#                plot_oosR2(path_df=path_df, var=var, value=value)
