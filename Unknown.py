# -*- coding: utf-8 -*-
import numpy as np
#import matplotlib.pyplot as plt
import cPickle

def unpickle(f):
    fo = open(f, 'rb')
    d = cPickle.load(fo)
    fo.close()
    return d

train_data = []
train_target = []
    
# 訓練データをロード
for i in range (1, 2):
    d = unpickle("cifar10/data_batch_%d"%(i))
    train_data.extend(d["data"])
    train_target.extend(d["labels"])
    
    print train_target[0]
    
    # テストデータをロード
    d = unpickle("cifar10/test_batch")
    test_data = d["data"]
    test_target = d["labels"]
    
    print test_target[0]
    
    # データはfloat32、ラベルはint32のndarrayに変換
    train_data = np.array(train_data, dtype=np.float32)
    train_target = np.array(train_target, dtype=np.int32)
    test_data = np.array(test_data, dtype=np.float32)
    test_target = np.array(test_target, dtype=np.int32)
    

