# -*- coding: utf-8 -*-
import time
import os, os.path
import numpy as np
from sklearn.cross_validation import train_test_split
import chainer
from chainer import cuda, optimizers
import chainer.functions as F
import chainer.links as L
from data_FinalTask import FinalTaskData
import cPickle

if __name__ == "__main__":
    
    batchsize = 4
    n_epoch = 8
    
    #データロード
    dataset = FinalTaskData()
    dataset.loadData()
    x_test, y_test = dataset.data_test["data"], dataset.data_train["labels"]
    N_test = y_test.size
    
    x_test = np.array(x_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.int32)
    
    gpu_flag = 0
    if gpu_flag >= 0:
        cuda.check_cuda_available()
    xp = cuda.cupy if gpu_flag >= 0 else np
    
    #モデルロード
    model = cPickle.load(open("FinalTask.pkl", "rb"))
    
    def forward (x_data, y_data, train = True):
        x, t = chainer.Variable(x_data), chainer.Variable(y_data)
        h = F.max_pooling_2d(F.relu(F.local_response_normalization(model.conv1(x))), 2, stride=2)
        h = F.max_pooling_2d(F.relu(F.local_response_normalization(model.conv2(h))), 2, stride=2)
        h = F.relu(model.conv3(h))
        h = F.relu(model.conv4(h))
        h = F.max_pooling_2d(F.relu(model.conv5(h)), 2, stride=2)
        h = F.dropout(F.relu(model.fc6(h)), train=train)
        h = F.dropout(F.relu(model.fc7(h)), train=train)
        y = model.fc8(h)
        
        if train:
            return F.softmax_cross_entropy(y, t)
        else:
            return F.accuracy(y, t) 
    
    if gpu_flag >= 0:
        cuda.get_device(gpu_flag).use()
        model.to_gpu()
        
    optimizer = optimizers.Adam()
    optimizer.setup(model)
    
    fp1 = open("accuracy_FinalTask.txt", "w")    
    fp1.write("Begin\nepoch\ttest_accuracy\n")
    
    #テスト
    for epoch in range(1, n_epoch + 1):
        sum_accuracy = 0
        for i in range(0, N_test, batchsize):
        	x_batch = xp.asarray(x_test[i: i + batchsize])
        	y_batch = xp.asarray(y_test[i: i + batchsize])

            
        	acc = forward(x_batch, y_batch, train=False)
        	sum_accuracy += float(acc.data) * len(y_batch)
        
        print "test accuracy: %f" % (sum_accuracy / N_test)


    fp1.write("%d\t%f\n" % (epoch, sum_accuracy / N_test))
    fp1.flush()
