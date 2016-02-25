# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 16:46:40 2016

@author: ubuntu
"""
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
         self.dump_name = u"dataset"
         self.image_size = 32
         
     def get_dir_list(self):
         tmp = os.listdir(self.data_dir_path)
         if tmp is None:
             return None
         return sorted([x for x in tmp if os.path.isdir(self.data_dir_path+x)])
