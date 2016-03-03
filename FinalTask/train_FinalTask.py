# -*- coding: utf-8 -*
import time
import os, os.path
import numpy as np
from sklearn.cross_validation import train_test_split
import chainer
from chainer import cuda, optimizers
import chainer.functions as F
from data_FinalTask import FinalTaskData
import cPickle

if __name__ == "__main__":
    
    gpu_flag = 0
    if gpu_flag >= 0:
        cuda.check_cuda_available()
    xp = cuda.cupy if gpu_flag >= 0 else np
    
    #初期設定
    batchsize = 4
    n_epoch = 8
    n_output = 4
    
    # データをロード
    print "data Loading..."
    dataset = FinalTaskData()
    dataset.DataCreate()
    dataset.loadData()
    x_train, y_train = dataset.data_train["data"], dataset.data_train["labels"]
    x_test, y_test = dataset.data_test["data"], dataset.data_train["labels"]
     # データはfloat32、ラベルはint32のndarrayに変換
    x_train = np.array(x_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    x_test = np.array(x_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.int32)
    
    
    N = y_train.size
    N_test = y_test.size
    print "data loaded."
    # 画像を (nsample, channel, height, width) の4次元テンソルに変換
    x_train = x_train.reshape((len(x_train), 3, 224, 224))
    x_test = x_test.reshape((len(x_test), 3, 224, 224))
    
    model = chainer.FunctionSet(
        conv1=F.Convolution2D(3,  96, 11, stride=3),
        conv2=F.Convolution2D(96, 256, 5, pad=2),
        conv3=F.Convolution2D(256, 384,  3, pad=0),
        conv4=F.Convolution2D(384, 384,  3, pad=0),
        conv5=F.Convolution2D(384, 256,  3, pad=0),
        fc6=F.Linear(9216, 4096),
        fc7=F.Linear(4096, 4096),
        fc8=F.Linear(4096, n_output)
        )
        
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
    
    fp2 = open("loss_FinalTask.txt", "w")
    fp2.write("Begin\nepoch\ttrain_loss\n")
    
    # 訓練ループ
    for epoch in range(1, n_epoch + 1):
        print "epoch: %d" % epoch

        perm = np.random.permutation(N)
        sum_loss = 0
        for i in range(0, N, batchsize):
            x_batch = xp.asarray(x_train[perm[i:i + batchsize]])
            y_batch = xp.asarray(y_train[perm[i:i + batchsize]])

            optimizer.zero_grads()
            loss = forward(x_batch, y_batch)
            loss.backward()
            optimizer.update()
            sum_loss += float(loss.data) * len(y_batch)
        sum_accuracy = 0
        for i in range(0, N_test, batchsize):
            x_batch = xp.asarray(x_test[i: i + batchsize])
            y_batch = xp.asarray(y_test[i: i + batchsize])

            
            acc = forward(x_batch, y_batch, train=False)
            sum_accuracy += float(acc.data) * len(y_batch)
        
        print "test accuracy: %f" % (sum_accuracy / N_test)
        
        print "train mean loss: %f" % (sum_loss / N)
        fp2.write("%d\t%f\n" % (epoch, sum_loss / N))
        fp2.flush()
        fp2.close()
        
    model.to_cpu()
    cPickle.dump(model, open("FinalTask.pkl", "wb"), -1)
    print "Finshing..."
