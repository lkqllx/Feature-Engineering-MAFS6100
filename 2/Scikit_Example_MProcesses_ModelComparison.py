 # -*- coding: utf-8 -*-
"""
This module is based on Prof.Chou's multi-processing program，
in order to calculate the out-of-sample R2 for stock 0050, 2330 under 5 models,
for different length of training period and prediction length.
Before running this module, please modify 
the length of prediction period in line 170, the stock in line 172, path in line 175, look-back period in line 179.
"""


import pandas as pd
import numpy as np
import os

#Scikit learn package
from sklearn import ensemble
from sklearn.linear_model import LinearRegression 
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

from multiprocessing import Pool
import time

def modelConstructionAndForecasting(numOfPastDays:int, numOfForwardDays:int, data:pd.DataFrame, numOfProcesses:int, modelResultDataFile:str):  
    Y_M_range = range(1,6)

    #Feature data
    featureColumns = ["GROUP1_0","GROUP1_1","GROUP1_2","GROUP1_3","GROUP1_4","GROUP5_0","GROUP5_1","GROUP5_2","GROUP5_3","GROUP5_4","GROUP6_0","GROUP6_1","GROUP6_2","GROUP6_3","GROUP6_4","GROUP7_1","GROUP7_2","GROUP7_3","GROUP7_4","GROUP7_5","GROUP15_1","GROUP15_2","GROUP15_3","GROUP15_4","GROUP15_5","GROUP11_1","GROUP11_2","GROUP8_1","GROUP8_2","GROUP8_3","GROUP8_4","GROUP8_5","GROUP9_0","GROUP9_1","GROUP9_2","GROUP9_3","GROUP9_4","GROUP10_1","GROUP10_2","GROUP10_3","GROUP10_4","GROUP10_5","GROUP12","GROUP13_1","GROUP13_2","GROUP13_3","GROUP13_4","GROUP13_5"]
    
    #Output data
#    columns1 = ['date', 'label', 'gbrt_isR2', 'linear_isR2', 'plsr_isR2', 'lasso_isR2', 'nn_isR2', \
#                                    'gbrt_oosR2', 'linear_oosR2', 'plsr_oosR2', 'lasso_oosR2', 'nn_oosR2']
    columns1 = ['training_start_date', 'testing_date', 'label', \
                                    'gbrt_oosR2', 'linear_oosR2', 'plsr_oosR2', 'lasso_oosR2', 'nn_oosR2']
    outputData = pd.DataFrame(columns=columns1)
    
    #Building gbrt model for each label
    gbrtModels = {}
    GBRTModels = {}
    for j in Y_M_range:
        gbrtModels['Y_M_{}'.format(str(j))] = ensemble.GradientBoostingRegressor()
    
    #Building linear regression model for each label
    linearModels = {}
    LINEARModels = {}
    for j in Y_M_range:
        linearModels['Y_M_{}'.format(str(j))] = LinearRegression(fit_intercept=True, normalize=True, copy_X=True)

    #Building partial least square regression model for each label
    plsrModels = {}
    PLSRModels = {}
    for j in Y_M_range:
        plsrModels['Y_M_{}'.format(str(j))] = PLSRegression(n_components = 8)

    #Building lassoCV model for each label
    lassoCVModels = {}
    LASSOCVModels = {}
    for j in Y_M_range:
        lassoCVModels['Y_M_{}'.format(str(j))] = LassoCV(eps=1e-2, n_alphas=100, alphas=None, fit_intercept=True, normalize=True, precompute='auto', max_iter=100000, tol=0.000001, copy_X=True, cv=5, verbose=False, n_jobs=1, positive=False, random_state=None, selection='cyclic')

    #Building Neural Network model for each label
    nnModels = {}
    NNModels = {}
    for j in Y_M_range:
        nnModels['Y_M_{}'.format(str(j))] = MLPRegressor(hidden_layer_sizes = (32, 32, 8), learning_rate = 'adaptive', early_stopping = True)
    
    date=data.date
    date_index=np.unique(date)
    date_num=date_index.size
    p=numOfPastDays
    
    for i in range(0,date_num-p-numOfForwardDays,1):
#    for i in range(date_num-p-numOfForwardDays):
        if i >= date_num-p-numOfForwardDays:  # 只做numOfForwardDays天的预测
            break
        print(">>>Processing training start date: " + date_index[i])
        
        # -------------------------------------------------------------------------------------
        #Prepare training and testing data for the current testing day
        trainingDates=[date_index[idx] for idx in range(i, i+p)]

        dataPd=data[data.date.isin(trainingDates)]
        #Use a dictionry to hold training label data
        y_training = {}
        for j in Y_M_range:
            y_training['Y_M_{}'.format(str(j))] = dataPd['Y_M_{}'.format(str(j))]
        
        #Hold feature data for training
        x_training = pd.DataFrame()
        for j in range(len(featureColumns)):
            x_training[featureColumns[j]] = dataPd[featureColumns[j]]

        ##Standardize for NN models specifically
        scaler = StandardScaler()
        ## Fit-Scale the training data
        scaler.fit(x_training)   ###Calculate mean and stdev for later scaling
        x_training_NN = scaler.transform(x_training)
            
        #timing the multi-Process prcessing
        t = time.time()   
        #Construction of models
        ##Establish a number of Processs for the calculation tasks
        pool = Pool(processes=numOfProcesses)

        for j in Y_M_range:
            GBRTModels['Y_M_{}'.format(str(j))] = pool.apply_async(gbrtModels['Y_M_{}'.format(str(j))].fit, args=(x_training, y_training['Y_M_{}'.format(str(j))]))
            LINEARModels['Y_M_{}'.format(str(j))] = pool.apply_async(linearModels['Y_M_{}'.format(str(j))].fit, args=(x_training, y_training['Y_M_{}'.format(str(j))]))
            PLSRModels['Y_M_{}'.format(str(j))] = pool.apply_async(plsrModels['Y_M_{}'.format(str(j))].fit, args=(x_training, y_training['Y_M_{}'.format(str(j))]))
            LASSOCVModels['Y_M_{}'.format(str(j))] = pool.apply_async(lassoCVModels['Y_M_{}'.format(str(j))].fit, args=(x_training, y_training['Y_M_{}'.format(str(j))]))
            NNModels['Y_M_{}'.format(str(j))] = pool.apply_async(nnModels['Y_M_{}'.format(str(j))].fit, args=(x_training_NN, y_training['Y_M_{}'.format(str(j))]))

        pool.close()
        pool.join()
        
        print(">>>Done with " + str(numOfProcesses) + " processes, and time taken is " + str(time.time() - t))
        
        # -------------------------------------------------------------------------------------   
#        for k in range(date_num-p-i):
        for k in range(numOfForwardDays):
            print(">>>Processing testing date: " + date_index[i+p+k])
            data1d=data[data.date==date_index[i+p+k]]         
            #Use a dictionry to hold testing label data
            y_testing = {}
            for j in Y_M_range:
                y_testing['Y_M_{}'.format(str(j))] = data1d['Y_M_{}'.format(str(j))]
    
            #Hold feature data for testing
            x_testing = pd.DataFrame()
            for j in range(len(featureColumns)):
                x_testing[featureColumns[j]] = data1d[featureColumns[j]]
                
            ## Only Scale the testing data
            x_testing_NN = scaler.transform(x_testing)    
            #Use a dictionry to hold predictions
            #y_prediction = {}
    
            '''
            #Use the models to predict
            for j in Y_M_range:
                y_prediction['Y_M_{}'.format(str(j))] = GBRTModels['Y_M_{}'.format(str(j))].get().predict(x_testing)
            '''
            
            #Now, we store modeling results
            for j in Y_M_range:
                oneLineModelResult = []
                oneLineModelResult.extend([str(date_index[i])])
                oneLineModelResult.extend([str(date_index[i+p+k])])
                oneLineModelResult.extend(['Y_M_{}'.format(str(j))])
                
                #Report in sample R2 results for each model one by one
    #            oneLineModelResult.extend([GBRTModels['Y_M_{}'.format(str(j))].get().score(x_training, y_training['Y_M_{}'.format(str(j))])])
    #            oneLineModelResult.extend([LINEARModels['Y_M_{}'.format(str(j))].get().score(x_training, y_training['Y_M_{}'.format(str(j))])])
    #            oneLineModelResult.extend([PLSRModels['Y_M_{}'.format(str(j))].get().score(x_training, y_training['Y_M_{}'.format(str(j))])])
    #            oneLineModelResult.extend([LASSOCVModels['Y_M_{}'.format(str(j))].get().score(x_training, y_training['Y_M_{}'.format(str(j))])])
    #            oneLineModelResult.extend([NNModels['Y_M_{}'.format(str(j))].get().score(x_training_NN, y_training['Y_M_{}'.format(str(j))])])
    
                oneLineModelResult.extend([GBRTModels['Y_M_{}'.format(str(j))].get().score(x_testing, y_testing['Y_M_{}'.format(str(j))])])            
                oneLineModelResult.extend([LINEARModels['Y_M_{}'.format(str(j))].get().score(x_testing, y_testing['Y_M_{}'.format(str(j))])])            
                oneLineModelResult.extend([PLSRModels['Y_M_{}'.format(str(j))].get().score(x_testing, y_testing['Y_M_{}'.format(str(j))])])            
                oneLineModelResult.extend([LASSOCVModels['Y_M_{}'.format(str(j))].get().score(x_testing, y_testing['Y_M_{}'.format(str(j))])])            
                oneLineModelResult.extend([NNModels['Y_M_{}'.format(str(j))].get().score(x_testing_NN, y_testing['Y_M_{}'.format(str(j))])])            
    
                outputData = pd.concat([outputData, pd.DataFrame(data = [oneLineModelResult], columns=columns1)], ignore_index=True)
        outputData.to_csv(modelResultDataFile + '_' + str(i) + '.csv')
    return outputData

if __name__ == '__main__':
    numOfForwardDays = 20 #Can be changed; must be no less than 1
    numOfProcesses = 6 #Change this number to measure calculation times
    featureData_list = ['step1_2330.csv.gz']
#    featureData_list = ['step1_2330.csv.gz']
    for ticker in featureData_list:
        featureDataDir = 'C://Users//EnhaoDaniel//111mafs6100C independent project feature engineering//HKUST_Summer2019_Taiwan//1top50Features//' + ticker
        df = pd.read_csv(featureDataDir, compression='gzip', index_col=0, header=0, sep=',', quotechar='"') 
        print(">>>Done loading inputData file.")        
        ticker = ticker[6:10]
        for numOfPastDays in [170]:
            print('numOfPastDays=' + str(numOfPastDays))
#            outputData = pd.concat([outputData, pd.DataFrame([[str(numOfPastDays),None,None,None,None,None,None,None]], columns=outputData.columns)], axis=0)
            outputfolder = 'C://Users//EnhaoDaniel//111mafs6100C independent project feature engineering//programs//Scikit_results//2330_numOfPastDays=' + str(numOfPastDays) + '//'
            os.makedirs(outputfolder)
            modelResultDataFile = outputfolder + 'modelStats_multiProcesses_' + ticker +'_all_' + str(numOfPastDays)
#            outputData = pd.concat([outputData, modelConstructionAndForecasting(numOfPastDays, numOfForwardDays, df, numOfProcesses, modelResultDataFile)], axis=0)
            modelConstructionAndForecasting(numOfPastDays, numOfForwardDays, df, numOfProcesses, modelResultDataFile)

#        for numOfForwardDays in range(1,21,1):
#            print('numOfForwardDays=' + str(numOfForwardDays))
#            modelResultDataFile = 'C://Users//EnhaoDaniel//111mafs6100C independent project feature engineering//programs//Scikit_results//modelStats_multiProcesses_' + ticker +'_all_' + str(numOfPastDays) + '.csv'
#            outputData = pd.concat([outputData, pd.DataFrame([[str(numOfForwardDays), None,None,None,None,None,None,None,None,None,None,None]], columns=outputData.columns)], axis=0)
#            outputData = pd.concat([outputData, modelConstructionAndForecasting(numOfPastDays, numOfForwardDays, df, numOfProcesses, modelResultDataFile)], axis=0)   
#            outputData.to_csv(modelResultDataFile)
