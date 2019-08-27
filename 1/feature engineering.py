import numpy as np
import time
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.linear_model import LassoCV
import pandas as pd
import time
import glob
import csv
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVR
from sklearn import datasets, linear_model, discriminant_analysis, model_selection
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR
import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv("step1_0050.csv",index_col=0)

#####Lasso Model
selected_feature=[]
coef=[]
r_int=[]
for k in range(130,191 ,10):
    r_sq = []
    for epoch in range(0, 233 - k):

        TRAIN_DURATION = k
        df['date_label'] = df.groupby(['date']).ngroup()
        train_start = epoch  # 0
        train_end = epoch + TRAIN_DURATION - 1  # 179
        test_end = epoch + TRAIN_DURATION  # 180
        train_data = df.loc[(df['date_label'] >= train_start) & (df['date_label'] <= train_end)]
        train_data.drop('date_label', axis=1, inplace=True)
        test_data = df.loc[(df['date_label'] > train_end) & (df['date_label'] <= test_end)]
        test_data.drop('date_label', axis=1, inplace=True)

        train_x = train_data.ix[:, 14:]
        train_y = train_data.ix[:, 6]
        test_x = test_data.ix[:, 14:]
        test_y = test_data.ix[:, 6]
        #LassoCV: 基于坐标下降法的Lasso交叉验证,这里使用30折交叉验证法选择最佳alpha
        print("第", epoch, "次", "区间是", k)
        print("使用坐标轴下降法计算参数正则化路径:")

        model=LassoCV(cv=30).fit(train_x, train_y)
        m_log_alphas = -np.log10(model.alphas_)
        alpha = model.alpha_
        lasso = Lasso(max_iter=10000, alpha=alpha)
        y_pred_lasso = lasso.fit(train_x, train_y).predict(test_x)
        a = r2_score(test_y, y_pred_lasso)
        print("r^2 is", a)
        r_sq.append(a)
        lasso.coef_ = sorted(lasso.coef_, reverse=True)
        coef_index = []
        for i in range(0, 48):
            if lasso.coef_[i] != 0:
                coef_index.append(i)
        x = train_x.ix[:, coef_index].columns.values
        length = len(coef_index)
        print("选中了", length, "个因子", "选中的因子是", x)
        selected_feature.append(x)
        coef.append(lasso.coef_)
    r_int.append(r_sq)
    pd.DataFrame(selected_feature).to_csv('lasso_2330_selected_feature_6' + str(k) + '.csv')
    pd.DataFrame(r_int).to_csv('lasso_2330_r_int_6' + str(k) + '.csv')

####Select 5 or 10 parameters (sklearn-select from model & Lasso)
selected_feature=[]
coef=[]
r_int=[]
for k in range(160,191 ,5):
    r_sq = []
    for epoch in range(0, 233 - k):

        TRAIN_DURATION = k
        df['date_label'] = df.groupby(['date']).ngroup()
        train_start = epoch  # 0
        train_end = epoch + TRAIN_DURATION - 1  # 179
        test_end = epoch + TRAIN_DURATION  # 180
        train_data = df.loc[(df['date_label'] >= train_start) & (df['date_label'] <= train_end)]
        train_data.drop('date_label', axis=1, inplace=True)
        test_data = df.loc[(df['date_label'] > train_end) & (df['date_label'] <= test_end)]
        test_data.drop('date_label', axis=1, inplace=True)

        train_x = train_data.ix[:, 14:]
        train_y = train_data.ix[:, 6]
        test_x = test_data.ix[:, 14:]
        test_y = test_data.ix[:, 6]
        #LassoCV: 基于坐标下降法的Lasso交叉验证,这里使用30折交叉验证法选择最佳alpha
        print("第", epoch, "次", "区间是", k)
        print("使用坐标轴下降法计算参数正则化路径:")
        clf = LassoCV(cv=5)
        # use max_features to choose parameters with better performance
        sfm = SelectFromModel(clf, max_features=10)
        sfm.fit(train_x, train_y)
        n_features = sfm.transform(train_x).shape[1]
        a = sfm.get_support(True)
        new_train_x=train_x.ix[:, a]
        new_test_x=test_x.ix[:, a]
        para_x=train_x.ix[:,a].columns
        print("选中的因子是", para_x)

        reg = LinearRegression().fit(new_train_x, train_y)
        y_pred_lasso=reg.predict(new_test_x)
        r2 = r2_score(test_y, y_pred_lasso)
        print("r^2 is", r2)
        r_sq.append(r2)
        selected_feature.append(para_x)
    r_int.append(r_sq)
    pd.DataFrame(selected_feature).to_csv('2330_5_selected_feature-10'+str(k)+'.csv')
    pd.DataFrame(r_int).to_csv('2330_5_r_int-10' + str(k) + '.csv')

########RFECV Model( sklearn RFE)
selected_feature=[]
coef=[]
r_int=[]
for k in range(160, 180, 10):
    r_sq = []
    for epoch in range(0, 233 - k):

        TRAIN_DURATION = k
        df['date_label'] = df.groupby(['date']).ngroup()
        train_start = epoch  # 0
        train_end = epoch + TRAIN_DURATION - 1  # 179
        test_end = epoch + TRAIN_DURATION  # 180
        train_data = df.loc[(df['date_label'] >= train_start) & (df['date_label'] <= train_end)]
        train_data.drop('date_label', axis=1, inplace=True)
        test_data = df.loc[(df['date_label'] > train_end) & (df['date_label'] <= test_end)]
        test_data.drop('date_label', axis=1, inplace=True)

        train_x = train_data.ix[:, 14:]
        train_y = train_data.ix[:, 2]
        test_x = test_data.ix[:, 14:]
        test_y = test_data.ix[:, 2]
        print("RFECV start：")
        print("第", epoch, "次", "区间是", k)
        estimator = SVR(kernel="linear")
        selector = RFECV(estimator, step=1, cv=5, min_features_to_select=10)
        selector = selector.fit(train_x, train_y)
        n_features = selector.transform(train_x).shape[1]
        a = selector.get_support(True)
        new_train_x = train_x.ix[:, a]
        new_test_x = test_x.ix[:, a]
        para_x = train_x.ix[:, a].columns
        print("选中的因子是", para_x)
        reg = LinearRegression().fit(new_train_x, train_y)
        y_pred_lasso = reg.predict(new_test_x)
        r2 = r2_score(test_y, y_pred_lasso)
        r_sq.append(r2)
        print("old_r^2 is", r2)
        selected_feature.append(para_x)
    r_int.append(r_sq)
    pd.DataFrame(selected_feature).to_csv('0050_1_RFECV_selected_feature-10' + str(k) + '.csv')
    pd.DataFrame(r_int).to_csv('0050_RFECV_**_r_int-10' + str(k) + '.csv')

####_______feature creation__________
###1.two of five selected features multifly randomly to create new features________


featureColumns = ["GROUP1_0","GROUP1_1","GROUP1_2","GROUP1_3","GROUP1_4","GROUP5_0","GROUP5_1","GROUP5_2","GROUP5_3","GROUP5_4","GROUP6_0","GROUP6_1","GROUP6_2","GROUP6_3","GROUP6_4","GROUP7_1","GROUP7_2","GROUP7_3","GROUP7_4","GROUP7_5","GROUP15_1","GROUP15_2","GROUP15_3","GROUP15_4","GROUP15_5","GROUP11_1","GROUP11_2","GROUP8_1","GROUP8_2","GROUP8_3","GROUP8_4","GROUP8_5","GROUP9_0","GROUP9_1","GROUP9_2","GROUP9_3","GROUP9_4","GROUP10_1","GROUP10_2","GROUP10_3","GROUP10_4","GROUP10_5","GROUP12","GROUP13_1","GROUP13_2","GROUP13_3","GROUP13_4","GROUP13_5"]
old_selected_feature=[]
new_selected_feature=[]
old_r_int=[]
new_r_int=[]
print("第", epoch, "次", "区间是", k)
for k in range(130, 191, 10):
    old_r_sq = []
    new_r_sq= []
    for epoch in range(0, 233 - k):

        TRAIN_DURATION = k
        df['date_label'] = df.groupby(['date']).ngroup()
        train_start = epoch  # 0
        train_end = epoch + TRAIN_DURATION - 1  # 179
        test_end = epoch + TRAIN_DURATION  # 180
        train_data = df.loc[(df['date_label'] >= train_start) & (df['date_label'] <= train_end)]
        train_data.drop('date_label', axis=1, inplace=True)
        test_data = df.loc[(df['date_label'] > train_end) & (df['date_label'] <= test_end)]
        test_data.drop('date_label', axis=1, inplace=True)

        #train_x = pd.DataFrame(Normalizer().fit_transform(train_data.ix[:, 14:]))
        #train_x.columns=featureColumns
        train_x=train_data.ix[:, 14:]
        train_y = train_data.ix[:, 6]
        #test_x = pd.DataFrame(Normalizer().fit_transform(test_data.ix[:, 14:]))
        #test_x.columns=featureColumns
        test_x=test_data.ix[:, 14:]
        test_y = test_data.ix[:, 6]
        # LassoCV: 基于坐标下降法的Lasso交叉验证,这里使用30折交叉验证法选择最佳alpha
        print("第", epoch, "次", "区间是", k)
        print("使用坐标轴下降法计算参数正则化路径:")
        max_features = 5
        # 基于训练数据，得到的模型的测试结果,这里使用的是坐标轴下降算法（coordinate descent）
        print("第", epoch, "次", "区间是", k)
        clf = LassoCV(cv=5)
        sfm = SelectFromModel(clf, max_features=max_features)
        sfm.fit(train_x, train_y)
        n_features = sfm.transform(train_x).shape[1]
        a = sfm.get_support(True)
        new_train_x = train_x.ix[:, a]
        new_test_x = test_x.ix[:, a]
        old_para_x = train_x.ix[:, a].columns

        print("选中的因子是", old_para_x)

        reg = LinearRegression().fit(new_train_x, train_y)
        y_pred_lasso = reg.predict(new_test_x)
        r2 = r2_score(test_y, y_pred_lasso)
        old_r_sq.append(r2)
        print("old_r^2 is", r2)
        print("creat new features")
        for i in range(0, max_features):
            for j in range(i+1,max_features):
                 new_train_x['a'+str(i)+str(j)]=new_train_x.ix[:,i]*new_train_x.ix[:,j]
                 new_test_x['a' + str(i) + str(j)] = new_test_x.ix[:, i] * new_test_x.ix[:, j]
                 print(i,j)

        sfm.fit(new_train_x, train_y)
        a = sfm.get_support(True)
        Nnew_train_x = new_train_x.ix[:, a]
        Nnew_test_x = new_test_x.ix[:, a]
        para_x = new_train_x.ix[:, a].columns
        print("选中的因子是", para_x)
        reg = LinearRegression().fit(Nnew_train_x, train_y)
        y_pred_lasso = reg.predict(Nnew_test_x)
        new_r2 = r2_score(test_y, y_pred_lasso)
        print("r^2 is", new_r2)
        new_r_sq.append(new_r2)
        new_selected_feature.append(para_x)
        old_selected_feature.append(old_para_x)
    old_r_int.append(old_r_sq)
    new_r_int.append(new_r_sq)
    pd.DataFrame(new_selected_feature).to_csv('new_2330_selected_feature' + str(k) + '.csv')
    pd.DataFrame(new_r_int).to_csv('new_2330_r_int' + str(k) + '.csv')

##### Use 11 GROUP_i_0 data to create new features. Use new features to do linear regression model,choose 10 features
# with the smallest p-value to make prediction.
# tried 3 methods:1. two of 11 parameters multiply randomly
#                 2. three of 11 parameters multiply randomly
#                 3. one parameter multiply with one of the other 10 parameters, the add itself as a new variable


group = [1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15]
selected_feature = []
coef = []
r_int = []
for k in range(160, 191, 10):
    r_sq = []
    for epoch in range(0, 233 - k):
        TRAIN_DURATION = k
        df['date_label'] = df.groupby(['date']).ngroup()
        train_start = epoch  # 0
        train_end = epoch + TRAIN_DURATION - 1  # 179
        test_end = epoch + TRAIN_DURATION  # 180
        train_data = df.loc[(df['date_label'] >= train_start) & (df['date_label'] <= train_end)]
        train_data.drop('date_label', axis=1, inplace=True)
        test_data = df.loc[(df['date_label'] > train_end) & (df['date_label'] <= test_end)]
        test_data.drop('date_label', axis=1, inplace=True)

        train_x = train_data.ix[:, 14:]
        train_y = train_data.ix[:, 2]
        test_x = test_data.ix[:, 14:]
        test_y = test_data.ix[:, 2]
        print("第", epoch, "次", "区间是", k)
        for i in range(0,9):
            for j in range(i+1,10):
                for h in range(j+1,11):
                    ####1st method:two of 11 parameters multiply randomly

                    # train_x['Group'+str(group[i])+'&'+str(group[j])+'_0']=train_x['GROUP{}_0'.format(str(group[i]))]*train_x['GROUP{}_0'.format(str(group[j]))]
                    # test_x['Group' + str(group[i]) + '&' + str(group[j]) + '_0'] =  test_x[
                    #                                                                     'GROUP{}_0'.format(str(group[i]))] * \
                    #                                                                 test_x[
                    #                                                                     'GROUP{}_0'.format(str(group[j]))]
                    ####2nd method:three of 11 parameters multiply randomly
                    train_x['Group'+str(group[i])+'&'+str(group[j])+str(group[h])+'_0']=train_x['GROUP{}_0'.format(str(group[i]))]*train_x['GROUP{}_0'.format(str(group[j]))]*train_x['GROUP{}_0'.format(str(group[h]))]
                    test_x['Group' + str(group[i]) + '&' + str(group[j]) + str(group[h])+'_0'] = test_x['GROUP{}_0'.format(str(group[i]))] * test_x['GROUP{}_0'.format(str(group[j]))]* test_x['GROUP{}_0'.format(str(group[h]))]

                    ###3rd method: one parameter multiply with one of the other 10 parameters, the add itself as a new variable
                    # train_x['Group'+str(group[i])+'&'+str(group[j])+'_0']=train_x['GROUP{}_0'.format(str(group[i]))]+train_x['GROUP{}_0'.format(str(group[i]))]*train_x['GROUP{}_0'.format(str(group[j]))]
                    # test_x['Group' + str(group[i]) + '&' + str(group[j]) + '_0'] = test_x[
                    #                                                                     'GROUP{}_0'.format(str(group[i]))] + \
                    #                                                                 test_x[
                    #                                                                     'GROUP{}_0'.format(str(group[i]))] * \
                    #                                                                 test_x[
                    #                                                                     'GROUP{}_0'.format(str(group[j]))]

        # print(i,j)
        train_x=train_x.ix[:,48:]
        test_x = test_x.ix[:, 48:]
        selector = SelectKBest(score_func=f_regression, k=5)
        selector.fit(train_x, train_y)
        a = selector.scores_.argsort()
        new_train_x=train_x.ix[:, a[-10:]]
        new_test_x = test_x.ix[:, a[-10:]]
        para_x=train_x.ix[:, a[-10:]].columns

        print(para_x)
        reg = LinearRegression().fit(new_train_x, train_y)
        y_pred_lasso = reg.predict(new_test_x)
        r2 = r2_score(test_y, y_pred_lasso)
        print("old_r^2 is", r2)
        r_sq.append(r2)
        selected_feature.append(para_x)
    r_int.append(r_sq)
    pd.DataFrame(selected_feature).to_csv('***_0050_1_scores_selected_feature-10' + str(k) + '.csv')
    pd.DataFrame(r_int).to_csv('***_0050__scores_r_int-10' + str(k) + '.csv')



