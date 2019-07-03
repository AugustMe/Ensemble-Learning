# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 10:06:24 2019

@author: ZQQ
"""

import numpy as np
from AdaBoost import AdaBoost
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load data：加载数据集
dataset = np.loadtxt('data.txt', delimiter=",")    
x = dataset[:, 0:8]
y = dataset[:, 8]

# prepare train data：分割得到训练集，测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# prepare test and train data
x_train=x_train.transpose()
y_train[y_train==1] = 1
y_train[y_train==0] = -1 # 二分类，标签为0的转换为-1

x_test=x_test.transpose()  
y_test[y_test == 1] = 1
y_test[y_test == 0] = -1

# train：开始训练
ada=AdaBoost(x_train, y_train) # 加载AdaBoost模型
ada.train(10) # 调用ada模型中的train

# predict
y_pred = ada.pred(x_test)
print ("total test", len(y_pred))
print ("true pred",  len(y_pred[y_pred == y_test]))
print ("acc", accuracy_score(y_test, y_pred))



### 改成调用函数形式
#def main():
#
#    # load data
#    dataset = np.loadtxt('data.txt', delimiter=",")    
#    x = dataset[:, 0:8]
#    y = dataset[:, 8]
#
#    # prepare train data
#    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
#
#    # prepare test and train data
#    x_train=x_train.transpose()
#    y_train[y_train==1] = 1
#    y_train[y_train==0] = -1
#
#    x_test=x_test.transpose()  
#    y_test[y_test == 1] = 1
#    y_test[y_test == 0] = -1
#
#    # train
#    ada=AdaBoost(x_train, y_train)
#    ada.train(10)
#
#    # predict
#    y_pred = ada.pred(x_test)
#    print ("total test", len(y_pred))
#    print ("true pred",  len(y_pred[y_pred == y_test]))
#    print ("acc", accuracy_score(y_test, y_pred))
#
#if __name__=='__main__':
#    main()