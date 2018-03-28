#!/usr/bin/python
# encoding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import label_binarize
from sklearn import metrics

## 设置属性防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

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
path = "D:\Git\TianChiBigDataCompetition\SearchAdPrediction\Python-Project\data\\round1_ijcai_18_train_20180301.txt"  # 数据文件路径
data = pd.read_csv(path, sep=' ')
# print data.head(1)
## 数据分割为特征属性和目标数据集合
X = data[names[0:-1]]
Y = data[names[-1:]]
print Y.head(1)

## 1. 缺失数据处理
## 使用Imputer给定缺省值，填充为均值(默认)
### 参数说明：
#### missing_values: 缺省值标识符号，默认为NaN,类型必须是数字
#### strategy: 填充缺省值的方式，默认为mean，可选mean、median、most_frequent
#### axis: 按照行还是按照列计算填充值，默认按照列计算(axis=0)
# X = X.replace("?", np.NAN)
# imputer = Imputer(missing_values="NaN")
# X = imputer.fit_transform(X, Y)
# print X[0]

## 2. 数据分割;数据划分为训练样本和测试样本
x_train,x_test,y_train,y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
print "训练样本数量:%d,特征属性数目:%d,目标属性数目:%d" % (x_train.shape[0],x_train.shape[1],y_train.shape[1])
print "测试样本数量:%d" % x_test.shape[0]

## 4. 归一化操作,可做可不做，随机森林不用
ss = MinMaxScaler()
x_train = ss.fit_transform(x_train, y_train)
x_test = ss.transform(x_test)