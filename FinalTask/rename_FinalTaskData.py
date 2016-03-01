# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 15:42:30 2016

@author: ubuntu
"""

import os
import sys
files = os.listdir('./')
for f in files:
    os.rename(f, f.replace(' ', ''))