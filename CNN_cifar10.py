# -*- coding: utf-8 -*
import time
import numpy as np
from sklearn.cross_validation import train_test_split
import chainer
from chainer import cuda, optimizers
import chainer.functions as F
import chainer.links as L

def unpickle(f):
    import cPickle
    fo = open(f, 'rb')
    d = cPickle.load(fo)
    fo.close()
    return d

def load_cifar10(datadir):
    train_data = []
    train_target = []
    
    # 訓練データをロード
    for i in range (1, 6):
        d = unpickle("%s/data_batch_%d"%(datadir, i))
        train_data.extend(d["data"])
        train_target.extend(d["labels"])
    
    # テストデータをロード
    d = unpickle("%s/test_batch"%(datadir))
    test_data = d["data"]
    test_target = d["labels"]
    
    # データはfloat32、ラベルはint32のndarrayに変換
    train_data = np.array(train_data, dtype=np.float32)
    train_target = np.array(train_target, dtype=np.int32)
    test_data = np.array(test_data, dtype=np.float32)
    test_target = np.array(test_target, dtype=np.int32)
    
    return train_data, test_data, train_target, test_target

if __name__ == "__main__":
    
    gpu_flag = 0
    if gpu_flag >= 0:
        cuda.check_cuda_available()
    xp = cuda.cupy if gpu_flag >= 0 else np
    
    
    #初期設定
    batchsize = 100
    n_epoch = 50
    
    # CIFAR-10データをロード
    print "load CIFAR-10 dataset"
    x_train, x_test, y_train, y_test = load_cifar10("cifar10")
    
    N = y_train.size
    N_test = y_test.size
    
    # 画像を (nsample, channel, height, width) の4次元テンソルに変換
    x_train = x_train.reshape((len(x_train), 3, 32, 32))
    x_test = x_test.reshape((len(x_test), 3, 32, 32))
    
    model = chainer.FunctionSet(conv1=F.Convolution2D(3,  32, 3, pad = 1),
        conv2=F.Convolution2D(32, 32,  3, pad=1),
        conv3=F.Convolution2D(32, 32,  3, pad=1),
        conv4=F.Convolution2D(32, 32,  3, pad=1),
        conv5=F.Convolution2D(32, 32,  3, pad=1),
        fc6=F.Linear(512, 512),
        fc7=F.Linear(512, 256),
        fc8=F.Linear(256, 10)
        )
    
    if gpu_flag >= 0:
    	model.
        
    def forward (x_data, y_data, train = True):
        x, t = chainer.Variable(x_data), chainer.Variable(y_data)
        h = F.max_pooling_2d(F.relu(F.local_response_normalization(model.conv1(x))), 2)
        h = F.max_pooling_2d(F.relu(F.local_response_normalization(model.conv2(h))), 2)
        h = F.relu(model.conv3(h))
        h = F.relu(model.conv4(h))
        h = F.max_pooling_2d(F.relu(model.conv5(h)), 2)
        h = F.dropout(F.relu(model.fc6(h)), train=train)
        h = F.dropout(F.relu(model.fc7(h)), train=train)
        y = model.fc8(h)
     
        if train:
        	return F.softmax_cross_entropy(y, t)
        else:
        	return F.accuracy(y, t)  
    model = cpickle.load(open(cifar10, 'rb'))
    if gpu_flag >= 0:
        cuda.get_device(gpu_flag).use()
        model.to_gpu()    
    optimizer = optimizers.Adam()
    optimizer.setup(model.collect_parameters())
    
    fp1 = open("accuracy_CNN.txt", "w")
    fp2 = open("loss_CNN.txt", "w")
    
    fp1.write("epoch\ttest_accuracy\n")
    fp2.write("epoch\ttrain_loss\n")
    
    # 訓練ループ
    start_time = time.clock()
    for epoch in range (1, n_epoch+1):
        print 'epoch', epoch
        perm = np.random.permutation(N)
        sum_loss = 0        
        for i in range (0, N, batchsize):
            x_batch = xp.asarray(x_train[perm[i:i+batchsize]])
            y_batch = xp.asarray(y_train[perm[i:i+batchsize]])
        
            optimizer.zero_grads()
            loss = forward(x_batch, y_batch)
            loss.backward()
            optimizer.update()
            sum_loss += float(loss.data) * batchsize
        
        print 'train mean loss = {}'.format(sum_loss/N)
        fp2.write("%d\t%f\n" % (epoch, sum_loss/N))
        
        
        
        #Test evaluation
        sum_acc = 0
    
        for i in range(0, N_test, batchsize):
            x_batch = xp.asarray(x_test[i:i+batchsize])
            y_batch = xp.asarray(y_test[i:i+batchsize])
        
            acc = forward(x_batch, y_batch, train = False)
            sum_acc +=float(acc.data)*batchsize
        print 'test accuracy = %f' % (sum_acc/N_test)
        fp1.write("%d\t%f\n" % (epoch, sum_acc / N_test))
        
        
        end_time = time.clock()
        print end_time - start_time
      	
    fp1.write("End\n")
    fp2.write("End\n")
    fp1.flush()
    fp2.flush()
    fp1.close()
    fp2.close()
    import cPickle
    model.to_cpu()
    cPickle.dump(model, open("cifar10.pkl", "wb"), -1)
    
        
	

                
           
