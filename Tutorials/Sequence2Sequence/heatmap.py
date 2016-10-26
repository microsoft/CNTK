# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 18:18:46 2016

@author: wdarling
"""

import numpy as np

X = np.array(shape=(7,6), dtype=np.float)

i=0
for line in open('attweightdata.txt').readlines():
    data = line.split()
    if len(data) > 1:
        X[i,:] = data
        i += 1
    else:
        break
    
