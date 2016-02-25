# -*- coding: utf-8 -*-
"""
Spyder Editor

This temporary script file is located here:
/home/ubuntu/.spyder2/.temp.py
"""
import time
import six.moves.cPickle as pickle
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions as F

class ImageNet(FunctionSet):
    def__init__(self, n_output):
        super(ImageNet, self).__init__(
            conv1 = F.Convolution2D(3, 32, 5),
            conv2 = F.Convolution2D(32, 32, 5),
            l3 = F.Linear(512, 512), 
            l4 = F.Linear(512, n_output)
            )
    def forward(self, x_data, y_data, train=True, gpu=-1):
        if gpu>=0:
            x_data = cuda.to_gpu(x_data)
            y_data = cuda.to_gpu(y_data)
            
        x, t = Variable(x_data), Variable(y_data)
        h = F.max_pooling_2d(F.relu(self.conv1(x)), ksize=2, stride=2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), ksize=3, stride=3)
        h = F.dropout(F.relu(self.l3(h)), train=train)
        y = self.l4(h)
        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

class CNN:
   def __init__(self, data, target, n_output, gpu=-1):
       self.model = ImageNet(n_output)
       self.model_name = 'cnn model'
       
       if gpu >= 0:
           self.model.to_gpu()
           
       self.gpu = gpu
       
       self.x_train,\
       self.x_test,\
       self.y_train,\
       self.y_test = train_test_split(data, target, test_size=0.1)
       
       self.n_train = len(self.y_train)
       self.n_test = len(self.y_test)
       
       self.optimizer = optimizers.Adam()
       self.optimizer.setup(self.model.collect_parameters())
       
def predict(self, x_data, gpu=-1):
        return self.model.predict(x_data, gpu)

def train_and_test(self, n_epoch=100, batchsize=100):

        epoch = 1
        best_accuracy = 0
        while epoch <= n_epoch:
            print 'epoch', epoch

            perm = np.random.permutation(self.n_train)
            sum_train_accuracy = 0
            sum_train_loss = 0
            for i in xrange(0, self.n_train, batchsize):
                x_batch = self.x_train[perm[i:i+batchsize]]
                y_batch = self.y_train[perm[i:i+batchsize]]

                real_batchsize = len(x_batch)

                self.optimizer.zero_grads()
                loss, acc = self.model.forward(x_batch, y_batch, train=True, gpu=self.gpu)
                loss.backward()
                self.optimizer.update()

                sum_train_loss += float(cuda.to_cpu(loss.data)) * real_batchsize
                sum_train_accuracy += float(cuda.to_cpu(acc.data)) * real_batchsize

            print 'train mean loss={}, accuracy={}'.format(sum_train_loss/self.n_train, sum_train_accuracy/self.n_train)

            # evaluation
            sum_test_accuracy = 0
            sum_test_loss = 0
            for i in xrange(0, self.n_test, batchsize):
                x_batch = self.x_test[i:i+batchsize]
                y_batch = self.y_test[i:i+batchsize]

                real_batchsize = len(x_batch)

                loss, acc = self.model.forward(x_batch, y_batch, train=False, gpu=self.gpu)

                sum_test_loss += float(cuda.to_cpu(loss.data)) * real_batchsize
                sum_test_accuracy += float(cuda.to_cpu(acc.data)) * real_batchsize

            print 'test mean loss={}, accuracy={}'.format(sum_test_loss/self.n_test, sum_test_accuracy/self.n_test)         

            epoch += 1

    def dump_model(self):
        self.model.to_cpu()
        pickle.dump(self.model, open(self.model_name, 'wb'), -1)

    def load_model(self):
        self.model = pickle.load(open(self.model_name,'rb'))
        if self.gpu >= 0:
            self.model.to_gpu()
        self.optimizer.setup(self.model.collect_parameters())

