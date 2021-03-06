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
    def __init__(self, n_output):
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
    def __init__(self, data, target, n_outputs, gpu=-1):

        self.model = ImageNet(n_outputs)
        self.model_name = 'cnn_model'

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

import os
import six.moves.cPickle as pickle
import numpy as np
import cv2 as cv

class AnimeFaceDataset:
    def __init__(self):
        self.data_dir_path = u"./animeface-character-dataset/thumb/"
        self.data = None
        self.target = None
        self.n_types_target = -1
        self.dump_name = u'dataset'
        self.image_size = 32

    def get_dir_list(self):
        tmp = os.listdir(self.data_dir_path)
        if tmp is None:
            return None
        return sorted([x for x in tmp if os.path.isdir(self.data_dir_path+x)])

    def get_class_id(self, fname):
        dir_list = self.get_dir_list()
        dir_name = filter(lambda x: x in fname, dir_list)
        return dir_list.index(dir_name[0])

    def load_data_target(self):
        if os.path.exists(self.dump_name):
            self.load_dataset()
        if self.target is None:
            dir_list = self.get_dir_list()
            ret = {}
            self.target = []
            target_name = []
            self.data = []
            for dir_name in dir_list:
                file_list = os.listdir(self.data_dir_path+dir_name)
                for file_name in file_list:
                    root, ext = os.path.splitext(file_name)
                    if ext == u'.png':
                        abs_name = self.data_dir_path+dir_name+'/'+file_name
                        # read class id i.e., target
                        class_id = self.get_class_id(abs_name)
                        self.target.append(class_id)
                        target_name.append(str(dir_name))
                        # read image i.e., data
                        image = cv.imread(abs_name)
                        image = cv.resize(image, (self.image_size, self.image_size))
                        image = image.transpose(2,0,1)
                        image = image/255.
                        self.data.append(image)

            self.index2name = {}
            for i in xrange(len(self.target)):
                self.index2name[self.target[i]] = target_name[i]

        self.data = np.array(self.data, np.float32)
        self.target = np.array(self.target, np.int32)

        self.dump_dataset()

    def get_n_types_target(self):
        if self.target is None:
            self.load_data_target()

        if self.n_types_target is not -1:
            return self.n_types_target

        tmp = {}
        for target in self.target:
            tmp[target] = 0
        return len(tmp)

    def dump_dataset(self):
        pickle.dump((self.data,self.target,self.index2name), open(self.dump_name, 'wb'), -1)

    def load_dataset(self):
        self.data, self.target, self.index2name = pickle.load(open(self.dump_name, 'rb'))