# -*- coding: utf-8 -*-
import os, os.path
import numpy as np
import matplotlib.pyplot as plt
import pprint
try:
    import cPickle as pickle 
except:
    import pickle

class FinalTaskData:
    
    def __init__(self):
        self.directory = u'/home/hidetoshi/MyProject/FistTestData/'
        self.file_dir = u'depth_data    0.txt'
        self.num_lines = 0
        self.num_files = 0
        self.merge_train = None
        self.merge_test = None
        self.labels_train = None
        self.labels_test = None
        self.dump_train = u'final_task_train.pkl' 
        self.dump_test = u'final_task_test.pkl' 
        self.data_train = None
        self.data_test = None
    
    def file_count(self):
        file_list = os.listdir(self.directory)
        for file_name in file_list :
            root, ext = os.path.splitext(file_name)
            if ext == u'.txt':
                self.num_files += 1
        return self.num_files
        """self.num_files = sum(1 for item in
                        os.listdir(self.directory)
                        if os.path.isfile(os.path.join(self.directory, item)))
           """             
        
    def line_count(self):
        f = open(self.directory + self.file_dir, 'r')
        self.num_lines = sum(1 for line in f)
        #return self.num_lines
    
    #Create an array from file in following order 
    #x-value, y-value, z-value of 1 image file
    def arrayCreate(self):
        #os.chdir("/home/ubuntu/MyProject")
        f = open(self.directory + self.file_dir, 'r')
        x, y, z = [], [], []
        for i in range(0, self.num_lines):
            x.append(f.readline())
            y.append(f.readline())
            z.append(f.readline())
            merge = np.array(float(x + y +z))
        return merge
    
    def labelCreate(self):
        #os.chdir("/home/ubuntu/MyProject")
        self.file_dir = u'正解ラベル.txt'
        f = open(self.directory + self.file_dir , 'r')
        for i in range (0, self.num_lines):
            for count in range (0,2):
                self.labels_train.append(int(f.readline()))
            self.labels_test.append (int(f.readline()))
                
    #Create a n x 150528 dimensions array from
    #1 x 150258 dimensions arrayCreate()
    def DataCreate(self):
        #print data_raw.shape
        self.line_count()
        self.file_count()
        for i in range (0,self.num_files - 2):
            if (i % 4 != 0):
                #print i
                self.file_dir = 'depth_data    %d.txt' %(i)
                a = self.arrayCreate()
                a = a.reshape(1,self.num_lines)
                self.merge_train = np.concatenateate((self.merge_train,a))
                #print data_raw.shape
            else:
                self.file_dir = 'depth_data    %d.txt' %(i)
                a = self.arrayCreate()
                a = a.reshape(1,self.num_lines)
                self.merge_test = np.concatenateate((self.merge_test,a))

        self.labelCreate()                
        self.pickleCreate()
    
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