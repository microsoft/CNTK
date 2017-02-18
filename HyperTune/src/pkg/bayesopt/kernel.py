#!/usr/bin/env python

# Copyright (c) Microsoft. All rights reserved.
# Licensed under custom Microsoft Research License
# See LICENSE.md file in the project root for full license information.

# Kernels for Gaussian Process


import os
import sys
import numpy as np
from ..util.distance import *
from ..util.randsample import *


#Main interface
class Kernel:
    def __init__(self, domain, kernel_type):
        if kernel_type == 'M52':
            self.actual_kernel = M52Kernel(domain)
        elif kernel_type == 'SE':
            self.actual_kernel = SEKernel(domain)
        else:
            raise ValueError("Unsupported kernel Type.")

    def set_scale(self, feature_scale):
        self.actual_kernel.set_scale(feature_scale)

    def compute_value(self, x, y):
        value = self.actual_kernel.compute_value(x, y)
        return value

    def compute_value_gradient(self, x, y):
        (value, grad) = self.actual_kernel.compute_value_gradient(x, y)
        return (value, grad)



#Abstract class
class Kernel_Abstract:

    def __init__(self, domain):
        self.domain = domain
        self.dimension = len(domain)
        self.distribution = Distance(domain)
        self.feature_scale = np.ones(self.dimension)

    def set_scale(self, feature_scale):
        self.feature_scale = np.array(feature_scale)

    def compute_distance(self, x, y):
        coordinate_dist = self.distribution.coordinate_dist(x, y)
        scaled_coord_distance = np.exp(self.feature_scale) * (coordinate_dist ** 2)
        dist = np.sqrt(scaled_coord_distance.sum())
        return (dist, coordinate_dist)

    def compute_value(self, x, y):
        value = 1.
        return value

    def compute_value_gradient(self, x, y):
        #Overwrite this function in real kernel
        value = 1.
        grad = np.ones(self.dimension)
        return (value, grad)



#Matern 5/2 kernel
class M52Kernel(Kernel_Abstract):

    def compute_value(self, x, y):
        (dist, coordinate_dist) = self.compute_distance(x, y)
        value_const = (1. + np.sqrt(5.)*dist + 5./3.*(dist**2)) 
        value = value_const * np.exp(- np.sqrt(5.) * dist)
        return value

    def compute_value_gradient(self, x, y):
        (dist, coordinate_dist) = self.compute_distance(x, y)
        value_const = (1. + np.sqrt(5.)*dist + 5./3.*(dist**2)) 
        value = value_const * np.exp(- np.sqrt(5.) * dist)
        grad_const = 5. * (1. + np.sqrt(5.)*dist) / (3. + 3.*np.sqrt(5.)*dist + 5.*(dist**2))
        grad = - grad_const * value * 0.5 * np.exp(self.feature_scale) * (coordinate_dist ** 2)
        return (value, grad)



#Squared exponential kernel
class SEKernel(Kernel_Abstract):

    def compute_value(self, x, y):
        (dist, coordinate_dist) = self.compute_distance(x, y)
        value = np.exp(-(dist**2) / 2)
        return value

    def compute_value_gradient(self, x, y):
        (dist, coordinate_dist) = self.compute_distance(x, y)
        value = np.exp(-(dist**2) / 2)
        grad = - value * 0.5 * np.exp(self.feature_scale) * (coordinate_dist ** 2)
        return (value, grad)






# For test purpose
if __name__ == '__main__':
    with open('src\\Examples\\Simple_hyper.py','r') as inf:
        HyperDict = eval(inf.read())
    cur_kernel = SEKernel(HyperDict)
    cur_sampler = RandomSampler(HyperDict)
    (value, grad) = cur_kernel.compute_value_gradient(cur_sampler.sample(), cur_sampler.sample())
    print(value)
    print(grad)


