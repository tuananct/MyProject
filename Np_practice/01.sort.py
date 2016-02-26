# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 17:06:40 2016

@author: ubuntu
"""

def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr)/2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

print quicksort(3, 10, 4, 1, 2, 6, 1, 7 )
