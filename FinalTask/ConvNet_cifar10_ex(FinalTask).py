# -*- coding: utf-8 -*-
import numpy as np
import chainer
from chainer import cuda
import chainer.functions as F
from chainer import optimizers
import time
from data_FinalTask import FinalTaskData
if __name__ == "__main__":
    gpu_flag = 0

    if gpu_flag >= 0:
        cuda.check_cuda_available()
    xp = cuda.cupy if gpu_flag >= 0 else np

    batchsize = 10
    n_epoch = 10

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
    
    print x_test.shape
	
"""    # 画像を (nsample, channel, height, width) の4次元テンソルに変換
    x_train = x_train.reshape((len(x_train), 3, 32, 32))
    x_test = x_test.reshape((len(x_test), 3, 32, 32))

    model = chainer.FunctionSet(conv1=F.Convolution2D(3, 32, 3, pad=0),
                                l1=F.Linear(7200, 512),
                                l2=F.Linear(512, 10))

    def forward(x_data, y_data, train=True):
        x, t = chainer.Variable(x_data), chainer.Variable(y_data)
        h = F.max_pooling_2d(F.relu(model.conv1(x)), 2)
        h = F.dropout(F.relu(model.l1(h)), train=train)
        y = model.l2(h)
        if train:
            return F.softmax_cross_entropy(y, t)
        else:
            return F.accuracy(y, t), y

    if gpu_flag >= 0:
        cuda.get_device(gpu_flag).use()
        model.to_gpu()

    optimizer = optimizers.Adam()
    optimizer.setup(model)

    fp1 = open("accuracy.txt", "w")
    fp2 = open("loss.txt", "w")

    fp1.write("epoch\ttest_accuracy\n")
    fp2.write("epoch\ttrain_loss\n")

    # 訓練ループ
    start_time = time.clock()
    for epoch in range(1, n_epoch + 1):
        print "epoch: %d" % epoch

        perm = np.random.permutation(N)
        sum_loss = 0
        for i in range(0, N, batchsize):
            x_batch = xp.asarray(X_train[perm[i:i + batchsize]])
            y_batch = xp.asarray(y_train[perm[i:i + batchsize]])

            optimizer.zero_grads()
            loss = forward(x_batch, y_batch)
            loss.backward()
            optimizer.update()
            sum_loss += float(loss.data) * len(y_batch)

        print "train mean loss: %f" % (sum_loss / N)
        fp2.write("%d\t%f\n" % (epoch, sum_loss / N))
        fp2.flush()

        sum_accuracy = 0
        for i in range(0, N_test, batchsize):
            x_batch = xp.asarray(X_test[i: i + 10])        
            y_batch = xp.asarray(y_test[i: i + 10])
            acc, pred = forward(x_batch, y_batch, train=False)
            sum_accuracy += float(acc.data) * len(y_batch)
        fp1.write("%d\t%f\n" % (epoch, sum_accuracy / N_test))
        fp1.flush()
         
#        	print "y_batch : ", 
#        	print y_batch
#        	print "acc ", 
#        	print acc
#        for idx in range (0, 10):
#        	xxx = xp.asarray(X_test[idx])
#        	h = F.max_pooling_2d(F.relu(model.conv1(xxx)), 2)
#        	h = F.dropout(F.relu(model.l1(h)), train=train)
#        	y = F.model.l2(h)
#        	print np.argmax(y.data)
#        print "test accuracy: %f" % (sum_accuracy / N_test)
        
    end_time = time.clock()
    print end_time - start_time
    
#    for idx in range (0, 10):
#    	x_batch = xp.asarray(X_test[idx])
#        y_batch = xp.asarray(y_test[idx])
#        acc, pred = forward(x_batch, y_batch, train = False)
#        print y_batch, np.argmax(pred.data)
        
    fp1.close()
    fp2.close()

    import cPickle
    model.to_cpu()
    cPickle.dump(model, open("cifar10.pkl", "wb"), -1)"""
