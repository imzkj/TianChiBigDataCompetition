#!/usr/bin/python
# encoding: utf-8

import  gc
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import OneHotEncoder
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from  sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from  sklearn.linear_model import LogisticRegressionCV
from scipy.sparse import csc_matrix
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

## 设置属性防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

path = "G:\TianChiBigDataCompetition\SearchAdPrediction\Python-Project\datas\\round1_ijcai_18_train_20180301.txt"  # # 数据文件路径

## 读取数据
### 加载数据后，通过data.columns获取names

names = [u'instance_id', u'item_id', u'item_category_list',
         u'item_property_list', u'item_brand_id', u'item_city_id',
         u'item_price_level', u'item_sales_level',
         u'item_collected_level', u'item_pv_level', u'user_id', u'user_gender_id',
         u'user_age_level', u'user_occupation_id',
         u'user_star_level', u'context_id',
         u'context_timestamp', u'context_page_id',
         u'predict_category_property', u'shop_id',
         u'shop_review_num_level', u'shop_review_positive_rate', u'shop_star_level',
         u'shop_score_service', u'shop_score_delivery', u'shop_score_description',
         u'is_trade']


def get_train_data():
    data = pd.read_csv(path, sep=' ')
    ## 数据分割为特征属性和目标数据集合
    X = data[names[0:-1]]
    Y = data[names[-1:]]
    print 'source date get finish...'
    M = X[[u'user_occupation_id', u'user_gender_id']]
    enc = OneHotEncoder(categorical_features='all')
    M_filter = M.loc[M[u'user_gender_id'] != -1].loc[M[u'user_occupation_id'] != -1]
    X_filter = X.loc[X[u'user_gender_id'] != -1].loc[X[u'user_occupation_id'] != -1]
    # print X_filter.index
    # for i in X_filter.index:
    del X
    del M
    gc.collect()
    #    print i
    print 'filter finish...'
    enc.fit(M_filter)
    occup_gender_matx = enc.transform(M_filter).toarray()
    del M_filter
    print 'gender finish'
    user_age_level = X_filter[u'user_age_level']
    user_age_matx = onehotInt(user_age_level)
    del user_age_level
    user_star_level = X_filter[u'user_star_level']
    user_star_matx = onehotInt(user_star_level)
    del user_star_level
    # print user_age_level
    # print user_age_level[0]
    item_category_list = X_filter[u'item_category_list']
    item_cate_matx = onehotString(item_category_list)
    del item_category_list
    print 'cate_matx finish...'
    # item_property_list = X_filter[u'item_property_list']
    # item_prop_matx = onehotString(item_property_list)
    # print 'prop_matx finish...'
    # pass

    item_brand_id = X_filter[u'item_brand_id']
    # item_brand_matx = onehotInt(item_brand_id)
    del item_brand_id
    print 'brand_matx finish...'
    item_city_id = X_filter[u'item_city_id']
    item_city_matx = onehotInt(item_city_id)
    del item_city_id
    print 'city_matx finish...'
    item_price_level = X_filter[u'item_price_level']
    item_sales_level = X_filter[u'item_sales_level']
    #print item_sales_level
    item_collected_level = X_filter[u'item_collected_level']
    item_pv_level = X_filter[u'item_pv_level']
    #print item_pv_level
    context_timestamp = X_filter[u'context_timestamp']
    print 'level finish...'
    context_page_id = X_filter[u'context_page_id']

    context_page_matx = onehotInt(context_page_id)
    context_page_id = None
    shop_review_num_level = X_filter[u'shop_review_num_level']
    #print shop_review_num_level
    shop_review_positive_rate = X_filter[u'shop_review_positive_rate']
    shop_star_level = X_filter[u'shop_star_level']
    shop_score_service = X_filter[u'shop_score_service']
    shop_score_delivery = X_filter[u'shop_score_delivery']
    shop_score_description = X_filter[u'shop_score_description']
    print 'all feature onehot finish...'
    # print item_cate_matx
    #del X_filter
    gc.collect()
    matx = np.hstack((occup_gender_matx, user_age_matx))
    occup_gender_matx = None
    user_age_level = None
    print '1....'
    matx = np.hstack((matx, user_star_matx))
    matx = np.hstack((matx, item_cate_matx))
    #matx = np.hstack((matx, item_brand_matx))
    print 'brand...'
    matx = np.hstack((matx, item_city_matx))
    price_arr = np.array(item_price_level.tolist())
    price_arr.shape = (len(price_arr),1)
    matx = np.hstack((matx, price_arr))

    sales_arr = np.array(item_sales_level.tolist())
    sales_arr.shape = (len(sales_arr), 1)
    matx = np.hstack((matx, sales_arr))

    collected_arr = np.array(item_collected_level.tolist())
    collected_arr.shape = (len(collected_arr), 1)
    matx = np.hstack((matx, collected_arr))

    pv_arr = np.array(item_pv_level.tolist())
    pv_arr.shape = (len(pv_arr), 1)
    matx = np.hstack((matx, pv_arr))

    timestamp_arr = np.array(context_timestamp.tolist())
    timestamp_arr.shape = (len(timestamp_arr), 1)
    matx = np.hstack((matx, timestamp_arr))
    matx = np.hstack((matx,context_page_matx))

    shop_review_num_arr = np.array(shop_review_num_level.tolist())
    shop_review_num_arr.shape = (len(shop_review_num_arr), 1)
    matx = np.hstack((matx, shop_review_num_arr))

    shop_review_positive_arr = np.array(shop_review_positive_rate.tolist())
    shop_review_positive_arr.shape = (len(shop_review_positive_arr), 1)
    matx = np.hstack((matx, shop_review_positive_arr))

    shop_star_arr = np.array(shop_star_level.tolist())
    shop_star_arr.shape = (len(shop_star_arr), 1)
    matx = np.hstack((matx, shop_star_arr))

    shop_score_service_arr = np.array(shop_score_service.tolist())
    shop_score_service_arr.shape = (len(shop_score_service_arr), 1)
    matx = np.hstack((matx, shop_score_service_arr))

    shop_score_delivery_arr = np.array(shop_score_delivery.tolist())
    shop_score_delivery_arr.shape = (len(shop_score_delivery_arr), 1)
    matx = np.hstack((matx, shop_score_delivery_arr))

    shop_score_description_arr = np.array(shop_score_description.tolist())
    shop_score_description_arr.shape = (len(shop_score_description_arr), 1)
    matx = np.hstack((matx, shop_score_description_arr))

    print matx.shape
    print 'over...'
    #sys.exit(-1)

    label = []

    for i in X_filter.index:
        label.append(Y[u'is_trade'][i])
    print len(label)
    print 'preprofdata ok...'
    X_train, X_test, Y_train, Y_test = train_test_split(matx, label, test_size=0.2)
    ## 数据正则化操作(归一化)
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train) ## 训练正则化模型，并将训练数据归一化操作
    X_test = ss.fit_transform(X_test) ## 使用训练好的模型对测试数据进行归一化操作
    print 'split finish...'
    # print len(X_train)
    # print len(X_test)
    logistic = LogisticRegressionCV(Cs=np.logspace(-4, 1, 50), fit_intercept=True, penalty='l2', solver='lbfgs',
                                    tol=0.01, multi_class='ovr')
    logistic.fit(X_train, Y_train)
    print 'training is finish '
    print logistic.predict_proba(X_test)
    ## Logistic算法效果输出
    logistic_r = logistic.score(X_train, Y_train)
    print "Logistic算法R值（准确率）：", logistic_r
    # print "Logistic算法稀疏化特征比率：%.2f%%" % (np.mean(logistic_r.coef_.ravel() == 0) * 100)
    # print "Logistic算法参数：", logistic_r.coef_
    # print "Logistic算法截距：", logistic_r.intercept_
    logistic_r_predict = logistic.predict_proba(X_test)
    print "log_loss value is : ", log_loss(Y_test, logistic_r_predict)


def onehotInt(series):
    val_set = set()
    for a in series:
        val_set.add(a)
    lst = list(val_set)
    matx = np.mat(np.zeros((len(series), len(lst))))
    i = 0
    for a in series:
        # vec = [0] * len(lst)
        index = lst.index(a)
        # vec[index] = 1
        matx[i, index] = 1
        i += 1
    return matx


def onehotString(stringSeries):
    val_set = set()
    print len(stringSeries)
    for src in stringSeries:
        vals = src.split(";")
        # print 'a1:',len(vals)
        for a in vals:
            val_set.add(a)

    lst = list(val_set)
    matx = np.mat(np.zeros((len(stringSeries), len(lst))))  # csc_matrix()
    val_set = None
    i = 0
    for src in stringSeries:
        vals = src.split(";")
        for a in vals:
            index = lst.index(a)
            matx[i, index] = 1
    return matx

    ## 2. 数据分割;数据划分为训练样本和测试样本
    # x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    # print "训练样本数量:%d,特征属性数目:%d,目标属性数目:%d" % (x_train.shape[0], x_train.shape[1], y_train.shape[1])
    # print "测试样本数量:%d" % x_test.shape[0]