# -*- coding: utf-8 -*
import time
import six.moves.cPickle as pickle
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
import chainer
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions as F
import chainer.links as L

def unpickle(f):
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

class ImageNet(FunctionSet):
    def __init__(self, n_outputs):
        super(ImageNet,self).__init__(
        conv1=L.Convolution2D(3,  32, 3, pad = 1),
        conv2=L.Convolution2D(32, 32,  3, pad=1),
        conv3=L.Convolution2D(32, 32,  3, pad=1),
        conv4=L.Convolution2D(32, 32,  3, pad=1),
        conv5=L.Convolution2D(32, 32,  3, pad=1),
        fc6=L.Linear(512, 512),
        fc7=L.Linear(512, 256),
        fc8=L.Linear(256, n_outputs)
        )
        self.train = True
        
    def forward (self, x_data, y_data, train = True):
        x, t = Variable(x_data), Variable(y_data)
        h = F.max_pooling_2d(F.relu(
            F.local_response_normalization(self.conv1(x))), 2)
        h = F.max_pooling_2d(F.relu(
            F.local_response_normalization(self.conv2(h))), 2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 2)
        h = F.dropout(F.relu(self.fc6(h)), train=self.train)
        h = F.dropout(F.relu(self.fc7(h)), train=self.train)
        y = self.fc8(h)

        self.loss = F.softmax_cross_entropy(y, t)
        self.accuracy = F.accuracy(y, t)
        return self.loss, self.accuracy

if __name__ == "__main__":
    batchsize = 100
    n_epoch = 20
    
    # CIFAR-10データをロード
    print "load CIFAR-10 dataset"
    x_train, x_test, y_train, y_test = load_cifar10("cifar10")
    N = y_train.size
    N_test = y_test.size
    
    # 画像を (nsample, channel, height, width) の4次元テンソルに変換
    x_train = x_train.reshape((len(x_train), 3, 32, 32))
    x_test = x_test.reshape((len(x_test), 3, 32, 32))
    
    model = AlexNet(10)  
    optimizer = optimizers.Adam()
    optimizer.setup(model)
    
    fp1 = open("accuracy.txt", "w")
    fp2 = open("loss.txt", "w")
    
    fp1.write("epoch\ttest_accuracy\n")
    fp2.write("epoch\ttrain_loss\n")
    # 訓練ループ
    start_time = time.clock()
    
    for epoch in xrange (1, n_epoch+1):
        print 'epoch', epoch
        perm = np.random.permutation(N)
        sum_train_loss = 0
        sum_train_acc = 0        
        for i in xrange (0, N, batchsize):
            x_batch = x_train[perm[i:i+batchsize]]
            y_batch = y_train[perm[i:i+batchsize]]
        
            optimizer.zero_grads()
            loss, acc = model.forward(x_batch, y_batch, train = True)
            loss.backward()
            optimizer.update()
            
            sum_train_loss += float(cuda.to_cpu(loss.data))*batchsize
            sum_train_acc += float(cuda.to_cpu(acc.data))*batchsize
        
        print 'train mean loss = {}, accuracy = {}'.format(sum_train_loss/N, sum_train_acc/N)
        fp2.write("%d\t%f\n" % (epoch, sum_loss/N))
        fp2.flush()
        
        
        #Test evaluation
        sum_test_loss = 0
        sum_test_acc = 0
    
        for i in xrange(0, N_test, batchsize):
            x_batch = x_test[i:i+batchsize]
            y_batch = y_test[i:i+batchsize]
        
            loss, acc = model.forward(x_batch, y_batch, train = False)
            sum_test_loss +=float(cuda.to_cpu(loss.data))*batchsize
            sum_test_acc +=float(cuda.to_cpu(acc.data))*batchsize
        print 'test mean loss = {}, accuracy = {}'.format(sum_test_loss/N_test, sum_test_acc/N_test)
        fp1.write("%d\t%f\n" % (epoch, sum_accuracy / N_test))
        fp1.flush()
        
        end_time = time.clock()
        print end_time - start_time
        fp1.close()
        fp2.close()

                
           
