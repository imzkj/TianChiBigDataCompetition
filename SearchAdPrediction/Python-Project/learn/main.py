#!/usr/bin/python
# encoding: utf-8

import FeatureProcess
import  numpy as np
from  sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix

def main():
    FeatureProcess.get_train_data()
    pass

if __name__ == '__main__':
    main()
    #a = np.mat(np.zeros((3,3)))
    #a[0,1] = 1
    #print a
    #a  = np.mat([1,0,2])
    #b = np.mat([0,0,1])
    #mm = np.hstack((a,[[1,0,0]]))
    #st = np.vstack((mm,mm))
    #print a
    #print st

