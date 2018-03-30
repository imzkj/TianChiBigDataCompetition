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
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from  sklearn.linear_model import LogisticRegressionCV,LogisticRegression
from scipy.sparse import csc_matrix,hstack,vstack,csr_matrix,lil_matrix
from sklearn.metrics import log_loss
import  pickle
import time
## 设置属性防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

path = "D:\Git\TianChiBigDataCompetition\SearchAdPrediction\Python-Project\data\\round1_ijcai_18_train_20180301.txt"  # # 数据文件路径
test_file = "D:\Git\TianChiBigDataCompetition\SearchAdPrediction\Python-Project\data\\round1_ijcai_18_test_a_20180301.txt"
model_file = "D:\Git\TianChiBigDataCompetition\SearchAdPrediction\Python-Project\data\logistic.mols"
feature_val_map_path = "D:\Git\TianChiBigDataCompetition\SearchAdPrediction\Python-Project\data\lfeature_val_map.mol"
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

def feature_encode(X_filter,feature_val_map,train = True):
    user_gender_id = X_filter[u'user_gender_id']
    if('user_gender_id' in feature_val_map):
        user_gender_lst = feature_val_map.get('user_gender_id')
    else:
        user_gender_lst = list(set(user_gender_id))
        feature_val_map['user_gender_id'] = user_gender_lst
    user_gender_matx = onehotInt(user_gender_id,user_gender_lst)
    del user_gender_id
    print user_gender_matx.shape
    matx = lil_matrix(user_gender_matx)
    del user_gender_matx

    user_occupation_id = X_filter[u'user_occupation_id']
    if ('user_occupation_id' in feature_val_map):
        user_occupation_lst = feature_val_map.get('user_occupation_id')
    else:
        user_occupation_lst = list(set(user_occupation_id))
        feature_val_map['user_occupation_id'] = user_occupation_lst
    user_occupation_matx = onehotInt(user_occupation_id, user_occupation_lst)
    del user_occupation_id
    print user_occupation_matx.shape
    matx = hstack((matx,user_occupation_matx))
    del user_occupation_matx
    #print matx.shape

    user_age_level = X_filter[u'user_age_level']
    if ('user_age_level' in feature_val_map):
        user_age_lst = feature_val_map.get('user_age_level')
    else:
        user_age_lst = list(set(user_age_level))
        feature_val_map['user_age_level'] = user_age_lst
    user_age_matx = onehotInt(user_age_level,user_age_lst)
    del user_age_level
    matx = hstack((matx, user_age_matx))
    del user_age_matx

    user_star_level = X_filter[u'user_star_level']
    if ('user_star_level' in feature_val_map):
        user_star_lst = feature_val_map.get('user_star_level')
    else:
        user_star_lst = list(set(user_star_level))
        feature_val_map['user_star_level'] = user_star_lst
    user_star_matx = onehotInt(user_star_level,user_star_lst)
    del user_star_level
    matx = hstack((matx, user_star_matx))
    del user_star_matx

    item_category_list = X_filter[u'item_category_list']
    if ('item_category_list' in feature_val_map):
        item_category_lst = feature_val_map.get('item_category_list')
    else:
        val_set = set()
        for src in item_category_list:
            vals = src.split(";")
            # print 'a1:',len(vals)
            for a in vals:
                val_set.add(a)
        item_category_lst = list(val_set)
        feature_val_map['item_category_list'] = item_category_lst
    item_cate_matx = onehotString(item_category_list,item_category_lst)
    del item_category_list
    matx = hstack((matx, item_cate_matx))
    del item_cate_matx
    gc.collect()
    """item_property_list = X_filter[u'item_property_list']
    if ('item_property_list' in feature_val_map):
        item_property_lst = feature_val_map.get('item_property_list')
    else:
        val_set = set()
        for src in item_property_list:
            vals = src.split(";")
            # print 'a1:',len(vals)
            for a in vals:
                val_set.add(a)
        item_property_lst = list(val_set)
        feature_val_map['item_property_list'] = item_category_lst
    item_prop_matx = onehotString(item_property_list,item_property_lst)
    del item_property_list
    matx = hstack((matx, item_prop_matx))
    del item_prop_matx
    # print 'prop_matx finish...'"""

    print 'brand start...'
    item_brand_id = X_filter[u'item_brand_id']
    if ('item_brand_id' in feature_val_map):
        item_brand_lst = feature_val_map.get('item_brand_id')
    else:
        item_brand_lst = list(set(item_brand_id))
        feature_val_map['item_brand_id'] = item_brand_lst
    print 'brand in...'
    item_brand_matx = onehotInt(item_brand_id,item_brand_lst)
    del item_brand_id
    matx = hstack((matx, item_brand_matx))
    del item_brand_matx
    print matx.shape

    item_city_id = X_filter[u'item_city_id']
    if ('item_city_id' in feature_val_map):
        item_city_lst = feature_val_map.get('item_city_id')
    else:
        item_city_lst = list(set(item_city_id))
        feature_val_map['item_city_id'] = item_city_lst
    item_city_matx = onehotInt(item_city_id,item_city_lst)
    del item_city_id
    matx = hstack((matx, item_city_matx))
    del item_city_matx
    print 'city_matx finish...'

    item_price_level = np.array(X_filter[u'item_price_level'].tolist())
    item_price_level.shape = ((len(item_price_level),1))
    matx = hstack((matx, item_price_level))

    item_sales_level = np.array(X_filter[u'item_sales_level'].tolist())
    item_sales_level.shape = ((len(item_sales_level), 1))
    matx = hstack((matx, item_sales_level))

    item_collected_level = np.array(X_filter[u'item_collected_level'].tolist())
    item_collected_level.shape = ((len(item_collected_level), 1))
    matx = hstack((matx, item_collected_level))

    item_pv_level = np.array(X_filter[u'item_pv_level'].tolist())
    item_pv_level.shape = ((len(item_pv_level), 1))
    matx = hstack((matx, item_pv_level))

    context_timestamp = np.array(X_filter[u'context_timestamp'].tolist())
    context_timestamp.shape = ((len(context_timestamp), 1))
    matx = hstack((matx, context_timestamp))

    print 'level finish...'

    context_page_id = X_filter[u'context_page_id']
    if ('context_page_id' in feature_val_map):
        context_page_lst = feature_val_map.get('context_page_id')
    else:
        context_page_lst = list(set(context_page_id))
        feature_val_map['context_page_id'] = context_page_lst
    context_page_matx = onehotInt(context_page_id,context_page_lst)
    del context_page_id
    matx = hstack((matx, context_page_matx))
    gc.collect()

    shop_review_num_level = np.array(X_filter[u'shop_review_num_level'].tolist())
    shop_review_num_level.shape = ((len(shop_review_num_level), 1))
    matx = hstack((matx, shop_review_num_level))

    # print shop_review_num_level
    shop_review_positive_rate = np.array(X_filter[u'shop_review_positive_rate'].tolist())
    shop_review_positive_rate.shape = ((len(shop_review_positive_rate), 1))
    matx = hstack((matx, shop_review_positive_rate))

    shop_star_level = np.array(X_filter[u'shop_star_level'].tolist())
    shop_star_level.shape = ((len(shop_star_level), 1))
    matx = hstack((matx, shop_star_level))

    shop_score_service = np.array(X_filter[u'shop_score_service'].tolist())
    shop_score_service.shape = ((len(shop_score_service), 1))
    matx = hstack((matx, shop_score_service))

    shop_score_delivery = np.array(X_filter[u'shop_score_delivery'].tolist())
    shop_score_delivery.shape = ((len(shop_score_delivery), 1))
    matx = hstack((matx, shop_score_delivery))

    shop_score_description = np.array(X_filter[u'shop_score_description'].tolist())
    shop_score_description.shape = ((len(shop_score_description), 1))
    matx = hstack((matx, shop_score_description))
    print 'all feature onehot finish...'
    gc.collect()
    print matx.shape
    print 'over...'
    # sys.exit(-1)
    if(train):
        write_file = open("D:\Git\TianChiBigDataCompetition\SearchAdPrediction\Python-Project\data\lfeature_val_map.mol",'w+')
        pickle.dump(feature_val_map,write_file)
    return matx


def get_train_data():
    data = pd.read_csv(path, sep=' ')
    ## 数据分割为特征属性和目标数据集合
    X = data[names[0:-1]]
    Y = data[names[-1:]]

    print 'source date get finish...'
    feature_val_map = {}
    matx = feature_encode(X,feature_val_map)
    label = []

    for i in X.index:
        label.append(Y[u'is_trade'][i])
    print len(label)
    print 'preprofdata ok...'
    evalute(matx,label)
    #training(matx,label)
    #predict(logistic)

def training(matx,label):
    ss = StandardScaler()
    X_train = ss.fit_transform(matx)
    logistic = LogisticRegression()
    logistic.fit(X_train, label)
    logistic_model = open(model_file, 'w+')
    pickle.dump(logistic,logistic_model)
    return logistic

def evalute(matx,label):
    X_train, X_test, Y_train, Y_test = train_test_split(matx, label, test_size=0.1)
    print 'split finish...'
    ss = StandardScaler(with_mean=False)
    X_train = ss.fit_transform(X_train)
    X_test = ss.fit_transform(X_test)
    # print len(X_test)
    logistic = LogisticRegressionCV(Cs=np.logspace(-4, 1, 50), fit_intercept=True, penalty='l2', solver='lbfgs',
                                    tol=0.01, multi_class='ovr')
    #logistic = LogisticRegression()
    logistic.fit(X_train, Y_train)
    print 'training is finish '
    print logistic.predict_proba(X_test)
    ## Logistic算法效果输出
    logistic_r = logistic.score(X_train, Y_train)
    print "Logistic算法R值（准确率）：", logistic_r
    #print "Logistic算法稀疏化特征比率：%.2f%%" % (np.mean(logistic_r.coef_.ravel() == 0) * 100)
    #print "Logistic算法参数：", logistic_r.coef_
    #print "Logistic算法截距：", logistic_r.intercept_
    logistic_r_predict = logistic.predict_proba(X_test)
    print "log_loss value is : ", log_loss(Y_test, logistic_r_predict)


def predict():
    X = pd.read_csv(test_file, sep=' ')
    ## 数据分割为特征属性和目标数据集合
    file_path = open(feature_val_map_path)
    feature_val_map = pickle.load(file_path)
    matx = feature_encode(X,feature_val_map)
    #ss = StandardScaler()
    #matx = ss.fit_transform(matx)
    print matx.shape
    print 'over...'
    model_path = open(model_file)
    logistic = pickle.load(model_path)
    print logistic.predict_proba([matx[0]])


def onehotInt(series,lst):
    #matx = np.mat(np.zeros((len(series), len(lst))))
    matx = lil_matrix((len(series), len(lst)))
    i = 0
    for a in series:
        # vec = [0] * len(lst)
        index = lst.index(a)
        # vec[index] = 1
        matx[i, index] = 1
        i += 1
    return matx


def onehotString(stringSeries,lst):
    #matx = np.mat(np.zeros((len(stringSeries), len(lst))))  # csc_matrix()
    matx = lil_matrix((len(stringSeries), len(lst)))
    i = 0
    for src in stringSeries:
        vals = src.split(";")
        for a in vals:
            index = lst.index(a)
            matx[i, index] = 1
    return matx

