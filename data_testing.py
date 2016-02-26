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
    
def line_count(directory):
    os.chdir("/home/ubuntu/MyProject")
    f = open(directory, 'r')
    num_lines = sum(1 for line in f)
    return num_lines

def file_count():
    directory = '/home/ubuntu/MyProject/FirstTestData'
    number_of_files = sum(1 for item in
                        os.listdir(directory)
                        if os.path.isfile(os.path.join(directory, item)))
    #print number_of_files - 4
    return number_of_files
    
#Create an array from file in following order 
#x-value, y-value, z-value of 1 image file
def arrayCreate(file_dir):
    os.chdir("/home/ubuntu/MyProject")
    f = open(file_dir, 'r')
    num_lines = line_count(file_dir)
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
    
def labelCreate():
    os.chdir("/home/ubuntu/MyProject")
    f = open('正解ラベル.txt', 'r')
    labels = []
    for i in range (0, line_count('正解ラベル.txt')):
        a = int(f.readline())
        labels.append(a)
    print len(labels)
    return labels
    
##Create a n x 150528 dimensions array from
#1 x 150258 dimensions arrayCreate()
def dataCreate():
    data_raw = arrayCreate('depth_data 0.txt')
    data_raw = data_raw.reshape(1,18)
    print data_raw.shape   
    for i in range (1,3):
        if (i % 4 != 0):
           #print i
            file_dir = 'depth_data %d.txt' %(i)
            a = arrayCreate(file_dir)
            a = a.reshape(1,18)
            data_raw = np.concatenate((data_raw,a))
            #print data_raw.shape
    return data
    
#Create pickle file from dict contained data and label    
def pickleCreate():           
    data_output = {'data' : dataCreate(), 'labels' : labelCreate()}
    output = open('final_task.pkl', 'wb')
    pickle.dump(data_output, output)
    output.close()
    
#Load the dataset    
def loadData():
    pkl_file = open('final_task.pkl', 'rb')
    data_load = pickle.load(pkl_file)
    pkl_file.close 
    return data_load
    
pickleCreate()
loadData()