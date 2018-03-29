#!/usr/bin/python
# encoding: utf-8

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

## 设置属性防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

path = "D:\Git\TianChiBigDataCompetition\SearchAdPrediction\Python-Project\data\\round1_ijcai_18_train_20180301.txt"  # # 数据文件路径

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
    print M_filter
    # print X_filter.index
    # for i in X_filter.index:
    #    print i
    print 'filter finish...'
    enc.fit(M_filter)
    occup_gender_matx = enc.transform(M_filter).toarray()
    print 'gender finish'
    user_age_level = X_filter[u'user_age_level']
    user_age_matx = onehotInt(user_age_level)
    user_star_level = X_filter[u'user_star_level']
    user_star_matx = onehotInt(user_star_level)
    # print user_age_level
    # print user_age_level[0]
    item_category_list = X_filter[u'item_category_list']
    item_cate_matx = onehotString(item_category_list)
    print 'cate_matx finish...'
    # item_property_list = X_filter[u'item_property_list']
    # item_prop_matx = onehotString(item_property_list)
    # print 'prop_matx finish...'
    # pass

    item_brand_id = X_filter[u'item_brand_id']
    item_brand_matx = onehotInt(item_brand_id)
    print 'brand_matx finish...'
    item_city_id = X_filter[u'item_city_id']
    item_city_matx = onehotInt(item_city_id)
    print 'city_matx finish...'
    item_price_level = X_filter[u'item_price_level']
    item_sales_level = X_filter[u'item_sales_level']
    item_collected_level = X_filter[u'item_collected_level']
    item_pv_level = X_filter[u'item_pv_level']

    context_timestamp = X_filter[u'context_timestamp']
    print 'level finish...'
    context_page_id = X_filter[u'context_page_id']

    context_page_matx = onehotInt(context_page_id)

    shop_review_num_level = X_filter[u'shop_review_num_level']
    shop_review_positive_rate = X_filter[u'shop_review_positive_rate']
    shop_star_level = X_filter[u'shop_star_level']
    shop_score_service = X_filter[u'shop_score_service']
    shop_score_delivery = X_filter[u'shop_score_delivery']
    shop_score_description = X_filter[u'shop_score_description']
    print 'all feature onehot finish...'
    # print item_cate_matx

    matx = np.hstack((occup_gender_matx, user_age_matx))
    matx = np.hstack((matx, user_star_matx))
    matx = np.hstack((matx, item_cate_matx))
    matx = np.hstack((matx, item_brand_matx))
    matx = np.hstack((matx, item_city_matx))
    matx = np.hstack((matx, item_city_matx))
    print matx.shape
    print 'over...'
    sys.exit(-1)

    matx = []
    m = 0
    for i, j in zip(X_filter.index, xrange(occup_gender_matx.shape[0])):
        # vec = occup_gender_matx[j].tolist()
        rmat = np.mat(occup_gender_matx[j])
        # print user_age_level[i]
        rmat = np.hstack((rmat, [[user_age_level[i]]]))
        # vec.append(user_age_level[i])
        # vec.append(user_star_level[i])
        rmat = np.hstack((rmat, [[user_star_level[i]]]))

        # vec.extend(item_cate_matx[j])
        rmat = np.hstack((rmat, item_cate_matx[j]))
        # vec.extend(item_prop_matx[j])
        # rmat = np.hstack((rmat, item_prop_matx[j]))
        # vec.extend(item_brand_matx[j])
        rmat = np.hstack((rmat, item_brand_matx[j]))
        # vec.extend(item_city_matx[j])
        rmat = np.hstack((rmat, item_city_matx[j]))

        # vec.append(item_price_level[i])
        rmat = np.hstack((rmat, [[item_price_level[i]]]))

        # vec.append(item_sales_level[i])
        rmat = np.hstack((rmat, [[item_sales_level[i]]]))

        # vec.append(item_collected_level[i])
        rmat = np.hstack((rmat, [[item_collected_level[i]]]))
        # vec.append(item_pv_level[i])
        rmat = np.hstack((rmat, [[item_pv_level[i]]]))
        # ec.append(context_timestamp[i])
        rmat = np.hstack((rmat, [[context_timestamp[i]]]))
        # vec.extend(context_page_matx[j])
        rmat = np.hstack((rmat, context_page_matx[j]))
        # vec.append(shop_review_num_level[i])
        rmat = np.hstack((rmat, [[shop_review_num_level[i]]]))

        # vec.append(shop_review_positive_rate[i])
        rmat = np.hstack((rmat, [[shop_review_positive_rate[i]]]))
        # vec.append(shop_star_level[i])
        rmat = np.hstack((rmat, [[shop_star_level[i]]]))
        # vec.append(shop_score_service[i])
        rmat = np.hstack((rmat, [[shop_score_service[i]]]))
        # vec.append(shop_score_delivery[i])
        rmat = np.hstack((rmat, [[shop_score_delivery[i]]]))
        # vec.append(shop_score_description[i])
        rmat = np.hstack((rmat, [[shop_score_description[i]]]))
        if m == 0:
            temp = rmat
        else:
            temp = np.vstack((temp, rmat))
        m += 1

    label = []

    for i in X_filter.index:
        label.append(Y[u'is_trade'][i])
    print len(label)
    print 'preprofdata ok...'
    X_train, X_test, Y_train, Y_test = train_test_split(matx, label, test_size=0.2)
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
    print "Logistic算法稀疏化特征比率：%.2f%%" % (np.mean(logistic_r.coef_.ravel() == 0) * 100)
    print "Logistic算法参数：", logistic_r.coef_
    print "Logistic算法截距：", logistic_r.intercept_
    logistic_r_predict = logistic.predict(X_test)
    print "log_loss value is : ", log_loss(Y_test, logistic_r_predict)


def onehotInt(series):
    val_set = set()
    for a in series:
        val_set.add(a)
    print 'int--offline:', len(val_set)
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

    print 'offline...', len(val_set)
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