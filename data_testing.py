# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 14:50:36 2016

@author: ubuntu
"""
import os, os.path
import numpy as np
import matplotlib.pyplot as plt
import pprint
try:
    import cPickle as pickle 
except:
    import pickle

def unpickle(f):
    fo = open(f, 'rb')
    d = cPickle.load(fo)
    fo.close()
    return d

def view():
    d = unpickle("cifar10/data_batch_1")
    i=0
    for key in d.iteritems():
        i = i + 1
    print i
    data = d["data"]
    labels = np.array (d["labels"])
    
def line_count():
    os.chdir("/home/ubuntu/MyProject")
    f = open('depth_data 0.txt', 'r')
    num_lines = sum(1 for line in f)
    return num_lines

def file_count():
    directory = '/home/ubuntu/MyProject/FirstTestData'
    number_of_files = sum(1 for item in
                        os.listdir(directory)
                        if os.path.isfile(os.path.join(directory, item)))
    print number_of_files - 4
    return number_of_file()
    
#Create an array from file in following order 
#x-value, y-value, z-value of 1 image file
def arrayCreate(file_dir):
    os.chdir("/home/ubuntu/MyProject")
    f = open(file_dir, 'r')
    num_lines = line_count()
    x, y, z = [], [], []
    for i in range(0, num_lines - 150522):
        k = float(f.readline())
        x.append(k)
        k = float(f.readline())
        y.append(k)
        k = float(f.readline())
        z.append(k)
    merge = np.array(x + y +z) 
    return merge

##Create a n x 150528 dimensions array from
#1 x 150258 dimensions arrayCreate()
def dataCreate():
    a = arrayCreate('depth_data 0.txt')
    a = a.reshape(1,18)
    print a.shape   
    for i in range (1,3):
        if (i % 4 != 0):
            print i
            file_dir = 'depth_data %d.txt' %(i)
            b = arrayCreate(file_dir)
            b = b.reshape(1,18)
            a = np.concatenate((a,b))
            print a.shape
    return {'data': a}
        
data1 = dataCreate()
data2 = pickle.dumps(data1)
data3 = pickle.loads(data2)
pprint.pprint(data3)