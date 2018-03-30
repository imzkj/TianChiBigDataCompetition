#!/usr/bin/python
# encoding: utf-8

import sys
import FeatureProcess
import  numpy as np
from  sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix,hstack,vstack,lil_matrix
from sklearn.preprocessing import  *
import pandas as pd

test_path = "D:\Git\TianChiBigDataCompetition\SearchAdPrediction\Python-Project\data\\test.txt"

def test():
    data = pd.read_csv(test_path, sep=' ')
    ## 数据分割为特征属性和目标数据集合
    X = data[FeatureProcess.names[0:-1]]
    Y = data[FeatureProcess.names[-1:]]
    entry = np.array(X[u'item_price_level'].tolist())
    print entry
    ss = MinMaxScaler()
    entry = ss.fit_transform([entry])
    print entry
    sys.exit(-1)
    entry.shape = (len(entry),1)
    print entry.shape
    A = csr_matrix(entry)
    print A.shape
    entry2 =  np.array(X[u'item_sales_level'])
    entry2.shape = (len(entry2),1)
    print entry2.shape
    A = np.hstack((A,entry2))
    print A.shape

def main():
    FeatureProcess.get_train_data()
    #FeatureProcess.predict()
    #test()

if __name__ == '__main__':
    main()
    #a = np.mat(np.zeros((4,3)))
    ##print a.shape
    #b = np.array([1,2,3,4])
    #b.shape = (len(b),1)
    #mm = b.T

    #print mm.shape
    #print np.hstack((a,b))
    #print a
    #a  = np.mat([1,0,2])
    #b = np.mat([0,0,1])
    #mm = np.hstack((a,[[1,0,0]]))
    #st = np.vstack((mm,mm))
    #print a
    #print st

