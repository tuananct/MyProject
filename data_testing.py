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
    
    def __init__(self):
        self.directory = u'/home/ubuntu/MyProject/FirstTestData'
        
    def file_count(self.directory):
        number_of_files = sum(1 for item in
                        os.listdir(directory)
                        if os.path.isfile(os.path.join(directory, item)))
        return number_of_files
    def get_file_list():
        self.file_list = os.listdir(self.directory)
   
    def line_count(self.directory):
        f = open(directory + 'depth_data 0.txt', 'r')
        self.num_lines = sum(1 for line in f)
    
    #Create an array from file in following order 
    #x-value, y-value, z-value of 1 image file
    def arrayCreate(self.file_dir):
        #os.chdir("/home/ubuntu/MyProject")
        f = open(self.file_dir, 'r')
        num_lines = line_count(self.file_dir)
        x, y, z = [], [], []
        for i in range(0, num_lines - 150522):
            k = float(f.readline())
            self.x.append(k)
            k = float(f.readline())
            self.y.append(k)
            k = float(f.readline())
            z.append(k)
            self.merge = np.array(x + y +z) 
    
    def labelCreate(self):
        #os.chdir("/home/ubuntu/MyProject")
        f = open(self.directory + '正解ラベル.txt', 'r')
        self.labels = []
        for i in range (0, line_count(self.directory + '正解ラベル.txt')):
            a = int(f.readline())
            self.labels.append(a)
    
    #Create a n x 150528 dimensions array from
    #1 x 150258 dimensions arrayCreate()
    def dataCreate(self):
        data_raw = arrayCreate(self.directory + 'depth_data 0.txt')
        data_raw = data_raw.reshape(1,18)
        #print data_raw.shape   
        for i in range (1,3):
            if (i % 4 != 0):
                #print i
                self.file_dir = 'depth_data %d.txt' %(i)
                a = arrayCreate(file_dir)
                a = a.reshape(1,18)
                self.data_raw = np.concatenate((data_raw,a))
                #print data_raw.shape
    
    #Create pickle file from dict contained data and label    
    def pickleCreate(self):           
        self.data_output = {'data' : self.data_raw, 'labels' : self.labels}
        self.output = open(self.directory + 'final_task.pkl', 'wb')
        pickle.dump(self.data_output, self.output)
        output.close()
    
    #Load the dataset    
    def loadData(self):
        pkl_file = open(self.directory + 'final_task.pkl', 'rb')
        self.data_load = pickle.load(pkl_file)
        pkl_file.close 
        