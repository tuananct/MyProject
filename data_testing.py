# -*- coding: utf-8 -*-
import os, os.path
import numpy as np
import matplotlib.pyplot as plt
import pprint
try:
    import cPickle as pickle 
except:
    import pickle

class FinalTaskData():
    
    def __init__(self.directory):
        #self.directory = u'/home/ubuntu/MyProject/FirstTestData'
    self.file_dir = u'depth_data 0.txt'
    self.dumb_train = u'final_task_train.pkl' 
    self.dumb_test = u'final_task_test.pkl' 
    self.num_files = 0
    self.num_lines = 0
    self.labels_train = []    
    self.labels_test = []
    self.merge_train = []
    self.merge_test = []
    self.data_train = {}
    self.data_test = {}
    
    def file_count(self.directory):
        file_list = os.listdir(self.directory)
        for file_name in file_list :
            root, ext = os.path.splitext(file_name)
            if ext == u'.txt':
                self.num_files += 1               
                #self.number_of_files = sum(1 for item in
                        #os.listdir(self.directory)
                        #if os.path.isfile(os.path.join(self.directory, item)))

    def line_count(self):
        f = open(self.directory + self.file_dir, 'r')
        self.num_lines = sum(1 for line in f)
    
    #Create an array from file in following order 
    #x-value, y-value, z-value of 1 image file
    def arrayCreate(self):
        #os.chdir("/home/ubuntu/MyProject")
        f = open(self.directory + self.file_dir, 'r')
        x, y, z = [], [], []
        for i in range(0, self.num_lines - 150522):
            x.append(float(f.readline()))
            y.append(float(f.readline()))
            z.append(float(f.readline()))
            self.merge = np.array(x + y +z) 
    
    def labelCreate(self):
        #os.chdir("/home/ubuntu/MyProject")
        self.file_dir = u'正解ラベル.txt'
        f = open(self.directory + self.file_dir , 'r')
        for i in range (0, self.num_lines):
            for count in range (0,2)
                self.labels_train.append(int(f.readline()))
            self.labels_test.append (int(f.readline()))
                
    #Create a n x 150528 dimensions array from
    #1 x 150258 dimensions arrayCreate()
    def trainDataCreate(self):
        #print data_raw.shape   
        for i in range (0,self.num_files - 2):
            if (i % 4 != 0):
                #print i
                self.file_dir = 'depth_data %d.txt' %(i)
                a = arrayCreate(self.directory + self.file_dir)
                a = a.reshape(1,self.num_lines)
                self.merge_train = np.concatenateate((self.merge_train,a))
                #print data_raw.shape
            else:
                self.file_dir = 'depth_data %d.txt' %(i)
                a = arrayCreate(self.directory + self.file_dir)
                a = a.reshape(1,self.num_lines)
                self.merge_test = np.concatenateate((self.merge_test,a))
    
    #Create pickle file from dict contained data and label    
    def pickleCreate(self):         
        self.data_train = {'data' : self.merge_train, 'labels' : self.labels_train}
        self.data_test = {'data' : self.merge_test, 'labels' : self.labels_test}
        pickle.dump(self.data_train, open(self.directory + self.dump_train , 'wb'), -1)
        pickle.dump(self.data_test, open(self.directory + self.dump_test , 'wb'), -1)

    #Load the dataset    
    def loadData(self):
        self.data_train = pickle.load(open(self.directory + self.dump_train, 'rb'))
        self.data_test = pickle.load(open(self.directory + self.dump_test, 'rb'))
        