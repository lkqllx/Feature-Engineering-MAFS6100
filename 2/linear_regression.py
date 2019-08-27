"""
This module is coded in the early stage of this project, in order to calculate the optimal back-end period.
It can also select the features with highest absolute t-values, get the frequency of top 20 features, coefficients, 
in-sample and out-of-sample R2 and win-ratio for all stocks, all labels, under linear Regression model.
"""
# %%
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats
import matplotlib.pyplot as plt



# %% returns the data we will use
def setup_path(datafdr_index, data_index):  # select data folder, data
    alldatafdr = 'C://Users//EnhaoDaniel//111mafs6100C independent project feature engineering//HKUST_Summer2019_Taiwan'  # all data folder
    #self.alldatafdr = r'C:\Users\EnhaoDaniel\111mafs6100C independent project feature engineering\HKUST_Summer2019_Taiwan_test'  # test data folder
    #self.alldatafdr = r'C:\Users\EnhaoDaniel\111mafs6100C independent project feature engineering\HKUST_Summer2019_Taiwan_view'  # test data folder
    os.chdir(alldatafdr)
    datafdrlist = os.listdir()  # data folder list

    datafdr = datafdrlist[datafdr_index]
    print('Working Folder: ' + str(datafdr))
    os.chdir(alldatafdr + '\\' + datafdr)
    curr_datalist = os.listdir()  # current data list
    data = curr_datalist[data_index]
    print('Data: ' + str(data))
    da = pd.read_csv(data)
    dates = np.unique(da['date'])
    dates_length = len(dates)
    return [da, dates_length]

# %% folder
#class linear_regression():
#def initiate(start_date_index=0, delta_day=10, test_date_length = 1):
def setup_data(da, start_date_index, delta_day, test_date_length):
    X = pd.DataFrame(columns=['GROUP1_0', 'GROUP1_1', 'GROUP1_2', 'GROUP1_3', 'GROUP1_4',
           'GROUP5_0', 'GROUP5_1', 'GROUP5_2', 'GROUP5_3', 'GROUP5_4', 'GROUP6_0',
           'GROUP6_1', 'GROUP6_2', 'GROUP6_3', 'GROUP6_4', 'GROUP7_1', 'GROUP7_2',
           'GROUP7_3', 'GROUP7_4', 'GROUP7_5', 'GROUP15_1', 'GROUP15_2',
           'GROUP15_3', 'GROUP15_4', 'GROUP15_5', 'GROUP11_1', 'GROUP11_2',
           'GROUP8_1', 'GROUP8_2', 'GROUP8_3', 'GROUP8_4', 'GROUP8_5', 'GROUP9_0',
           'GROUP9_1', 'GROUP9_2', 'GROUP9_3', 'GROUP9_4', 'GROUP10_1',
           'GROUP10_2', 'GROUP10_3', 'GROUP10_4', 'GROUP10_5', 'GROUP12',
           'GROUP13_1', 'GROUP13_2', 'GROUP13_3', 'GROUP13_4', 'GROUP13_5'])
    Y = pd.DataFrame(columns=['Y_M_1', 'Y_M_2', 'Y_M_3', 'Y_M_4', 'Y_M_5'])
    X_test = X
    Y_test = Y
    
    dates = np.unique(da['date'])
    start_date = dates[start_date_index]
    end_date = dates[start_date_index + delta_day]
    test_end_date = dates[start_date_index + delta_day + test_date_length]
#    da['date'] = pd.to_datetime(da['date'])  
    da_train = da[(da['date'] >= start_date) & (da['date'] < end_date)]  
    X = pd.concat([X, da_train.iloc[:, 15:63]], axis=0)
    Y = pd.concat([Y, da_train.iloc[:, 3:8]], axis=0)
    da_test = da[(da['date'] >= end_date) & (da['date'] < test_end_date)]
    X_test = pd.concat([X_test, da_test.iloc[:, 15:63]], axis=0)
    Y_test = pd.concat([Y_test, da_test.iloc[:, 3:8]], axis=0)
                    
    X = X.reset_index(drop=True).astype(float)
    Y = Y.reset_index(drop=True).astype(float)
    X_test = X_test.reset_index(drop=True).astype(float)
    Y_test = Y_test.reset_index(drop=True).astype(float)
    return [X, Y, X_test, Y_test]

# %% linear regression
def lin_reg(X,y,X_test,y_test):
    lm = LinearRegression()
    lm.fit(X,y)
    
    y_pred = lm.predict(X)    
    y_mean = y.mean()
    SSE = sum((y-y_pred)**2)
    SST = sum((y-y_mean)**2)
    R_sq = 1 - SSE/SST
    win_ratio = pd.Series(np.sign(y_pred)).eq(pd.Series(np.sign(y))).mean()
    
    y_pred_test = lm.predict(X_test)
    y_mean_test = y_test.mean()
    SSE_test = sum((y_test - y_pred_test)**2)
    SST_test = sum((y_test - y_mean_test)**2)
    R_sq_test = 1 - SSE_test/SST_test
    win_ratio_test = pd.Series(np.sign(y_pred_test)).eq(pd.Series(np.sign(y_test))).mean()
    
    params = np.append(lm.intercept_, lm.coef_)
    newX = pd.DataFrame({"Constant":np.ones(len(X))}).join(pd.DataFrame(X))
    MSE = SSE/(len(newX)-len(newX.columns))
    # 算出各个beta的student t
    var_b = MSE*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
    sd_b = np.sqrt(var_b)
    ts_b = params/ sd_b
    abs_ts_b = abs(ts_b)
    
    p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newX)-1))) for i in ts_b]
    variable_name = np.concatenate((['intercept'], list(X.columns)), axis=0)
    
    myResult = pd.DataFrame()
    myResult['features'],myResult['Coefficients'],myResult['absolute t values'],myResult['p values'] = [variable_name, params,abs_ts_b,p_values]

    myCoef = myResult.iloc[:,[0,1]]
    myAbs_ts_b = myResult.iloc[1:,[0,2]]
    myP_values = myResult.iloc[1:,[0,3]]
    return [myCoef, myAbs_ts_b, myP_values, R_sq, win_ratio, R_sq_test, win_ratio_test]

    
# %%
def record_results(mode, datafdr_index, data_index, delta_day, test_date_length):

    [da, dates_length] = setup_path(datafdr_index, data_index)
    
    coefficient_dict={}
    myAbs_ts_b_dict={}
    myP_values_dict={}
    R_sq_dict={}
    win_ratio_dict={}
    R_sq_test_dict={}
    win_ratio_test_dict={}
    for idx in range(5):
        print('index of y label: ' + str(idx))
        coefficient_list=[]
        myAbs_ts_b_list=[]
        myP_values_list=[]
        R_sq_list=[]
        win_ratio_list=[]
        R_sq_test_list=[]
        win_ratio_test_list=[]
        if mode == 'discrete':
            rg = range(0, dates_length-1-delta_day, delta_day)
        elif mode == 'continuous':
            rg = range(0, dates_length-1-delta_day, 1)
        else:
            print('mode input error!')
            break
        for start_date_index in rg:
            if start_date_index % 30 == 0:
                print('starting date index: ' + str(start_date_index))
            [X, Y, X_test, Y_test] = setup_data(da, start_date_index, delta_day, test_date_length)
            [myCoef, myAbs_ts_b, myP_values, R_sq, win_ratio, R_sq_test, win_ratio_test] = lin_reg(X, Y.iloc[:, idx], X_test, Y_test.iloc[:, idx])
            
            coefficient_list.append(myCoef)
            myAbs_ts_b_list.append(myAbs_ts_b)
            myP_values_list.append(myP_values)
            
            R_sq_list.append(R_sq)
            win_ratio_list.append(win_ratio)
            R_sq_test_list.append(R_sq_test)    
            win_ratio_test_list.append(win_ratio_test)
            
        coefficient_dict['Y_{}'.format(idx)] = coefficient_list
        myAbs_ts_b_dict['Y_{}'.format(idx)] = myAbs_ts_b_list
        myP_values_dict['Y_{}'.format(idx)] = myP_values_list
        
        R_sq_dict['Y_{}'.format(idx)] = R_sq_list
        win_ratio_dict['Y_{}'.format(idx)] = win_ratio_list
        R_sq_test_dict['Y_{}'.format(idx)] = R_sq_test_list
        win_ratio_test_dict['Y_{}'.format(idx)] = win_ratio_test_list
    return [coefficient_dict, myAbs_ts_b_dict, myP_values_dict, R_sq_dict, win_ratio_dict, R_sq_test_dict, win_ratio_test_dict]


# %%
def sort_results(y_index=0):
    myAbs_ts_b_list = myAbs_ts_b_dict['Y_{}'.format(y_index)]
    myP_values_list = myP_values_dict['Y_{}'.format(y_index)]
    

    myAbs_ts_b_sum = myAbs_ts_b_list[0].iloc[:,1]
    myP_values_logsum = np.log(myP_values_list[0].iloc[:,1])
    freq_by_t_df = myAbs_ts_b_list[0].sort_values(by=['absolute t values'], ascending=False).iloc[:20, 0].value_counts()
    
    for i in range(1,len(myAbs_ts_b_list)):
        myAbs_ts_b_sum += myAbs_ts_b_list[i].iloc[:,1]
        myP_values_logsum += np.log(myP_values_list[i].iloc[:,1])
        freq_by_t_df = pd.concat([freq_by_t_df, 
                                  myAbs_ts_b_list[i].sort_values(by=['absolute t values'], ascending=False).iloc[:20, 0].value_counts()], axis=1)
        
    myAbs_ts_b_sum = pd.concat([myAbs_ts_b_list[0].iloc[:,0], myAbs_ts_b_sum], axis=1).sort_values(by=['absolute t values'], ascending=False)
    myP_values_logsum = pd.concat([myP_values_list[0].iloc[:,0], myP_values_logsum], axis=1).sort_values(by=['p values'], ascending=True)
    freq_by_t_df =  freq_by_t_df.sum(axis=1, skipna=True).sort_values(axis=0, ascending=False)

    return [myAbs_ts_b_sum, myP_values_logsum, freq_by_t_df]


      
# %%
def plot(x_cord = list(range(15, 200, 5))):
    Rsqt_cord = {}
    wint_cord = {}
    for i in range(5):
        Rsqt_cord[i] = []
        wint_cord[i] = []
    for j in x_cord:
        for i in range(5):
            Rsqt_cord[i].append(np.mean(results_dict[j][5]['Y_{}'.format(i)]))
            wint_cord[i].append(np.mean(results_dict[j][6]['Y_{}'.format(i)]))
    for i in range(5):
#        plt.plot(x_cord, Rsqt_cord[i])
        plt.plot(x_cord, wint_cord[i])
plot()

# %% 
if __name__ == '__main__':
    # %% get all the dictionaries (results)
    datafdr_index, data_index = 0, 23
    #mode = 'discrete'
    #delta_day, test_date_length = 10, 1
    mode = 'continuous'
    test_date_length = 1
    
    results_dict={}
    for delta_day in range(5, 200, 5):
        print('processing delta_day ' + str(delta_day))
        results_dict[delta_day] = record_results(mode, datafdr_index, data_index, delta_day, test_date_length)
        
    # %% sum and get frequencies
    [myAbs_ts_b_sum, myP_values_logsum, freq_by_t_df] = sort_results(y_index=0)
    myAbs_ts_b_sum_top20 = myAbs_ts_b_sum.iloc[:20,:]#.sort_values(by=['features'], ascending=True)
    myP_values_logsum_top20 = myP_values_logsum.iloc[:20,:]#.sort_values(by=['features'], ascending=True)
    
    # %% illustrate results
    #myAbs_ts_b_sum_top20
    #myP_values_logsum_top20
    freq_by_t_df
    #plt.plot(myAbs_ts_b_sum_top20.iloc[:,1].reset_index(drop=True),'ro')
    #myAbs_ts_b_dict['Y_0'][0].sort_values(by=['absolute t values'],ascending=False).reset_index(drop=True)
    #plt.plot(myAbs_ts_b_dict['Y_0'][0].sort_values(by=['absolute t values'],ascending=False).iloc[:,1].reset_index(drop=True),'ro')
    
    #myP_values_dict['Y_0'][0].sort_values(by=['p values'],ascending=True)
    #plt.plot(myP_values_dict['Y_0'][0].sort_values(by=['p values'],ascending=True).iloc[:20,1].reset_index(drop=True), 'ro')
    

