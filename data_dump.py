# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 15:24:21 2016

@author: ubuntu
"""

try:
    import cPickle as pickle 
except:
    import pickle
import pprint
data1 = [{'a':'A', 'b':2, 'c': 3.0}]
print 'BEFORE: ', 
print(data)

data2 = pickle.dumps(data1)

data3 = pickle.loads(data2)
print 'AFTER: ',
pprint.pprint(data3)

