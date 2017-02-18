#!/usr/bin/env python

# Copyright (c) Microsoft. All rights reserved.
# Licensed under custom Microsoft Research License
# See LICENSE.md file in the project root for full license information.

import os
import sys
import numpy as np
from .bayesopt import *
from .gauprocess import *
from .kernel import *
from ..util.distance import *
from ..util.randsample import *
import pdb


class BayesOptimizationAdd(BayesOptimization):

    def __init__(self, domains, model_spec = {}):
        self.sampler = RandomSamplerAdd(domains)
        #Hyperparameter in model_spec:
        self.read_model_spec(model_spec)
        self.bayesian_opt_model = GaussianProcessAdd(domains, self.kernel_type, self.base_noise)
        self.num_additive = len(domains)

    def select_best(self, exclude_list):
        #Use this after fit
        x_best_list = []
        for additive_index in range(self.num_additive):
            score_list = []
            for i in range(self.sample_size):
                x_new = self.sampler.sample_composite(additive_index)
                (mean, std) = self.bayesian_opt_model.predict_composite(x_new, additive_index)
                criterion = self.computeUCB(mean, std, self.sampler.Dims[additive_index])
                score_list.append((x_new, criterion))

            min_val = min([score[1] for score in score_list])
            ind_max = np.argmax([score[1] for score in score_list])
            (x_best, criterion_best) = score_list[ind_max]
            while x_best in exclude_list:
                score_list[ind_max][1] = min_val  # move to end
                ind_max = np.argmax([score[1] for score in score_list])
                (x_best, criterion_best) = score_list[ind_max]
            x_best_list.append(x_best)
        return x_best_list



class GaussianProcessAdd(GaussianProcess):

    def __init__(self, domains, kernel_type, base_noise):
        self.dimension = sum([len(domain) for domain in domains])
        self.kernel = KernelAdd(domains, kernel_type)
        self.base_noise = base_noise
        self.initialize()

    
    def predict_composite(self, x_slice, additive_index):
        # use fit before predict
        # return mean and std
        self_kernel_value = self.kernel.compute_value_composite(x_slice, x_slice, additive_index)
        mutual_kernel_value = np.array([self.kernel.compute_value_composite(x_slice, xp[additive_index], additive_index) \
            for xp in self.xlist])
        inv_kernel_matrix = self.inverse_kernel_matrix.dot(mutual_kernel_value)

        mean = inv_kernel_matrix.dot(self.y_minus_mean)
        var = self.kernel_amplifier * (self_kernel_value - inv_kernel_matrix.dot(mutual_kernel_value))
        std = np.sqrt(var)
        return (mean, std)




class KernelAdd(Kernel_Abstract):
    def __init__(self, domains, ktype):
        # Domains for a list of domain to denote the additive group
        self.kernel_list = [Kernel(domain, ktype) for domain in domains]

        self.num_additive = len(domains)
        self.dimensions = [len(domain) for domain in domains]
        self.split_indices = [self.dimensions[0]]
        for i in range(1, self.num_additive-1):
            self.split_indices.append(self.split_indices[i - 1] + self.dimensions[i])
    

    def slice(self, x):
        return np.split(x, self.split_indices)
            

    def set_scale(self, feature_scale):
        feature_scales = self.slice(feature_scale)
        for i in range(self.num_additive):
            self.kernel_list[i].set_scale(feature_scales[i])


    def compute_value_gradient(self, x, y):
        value_all = 0
        grad_all = np.empty(0)

        for i in range(self.num_additive):
            (value, grad) = self.kernel_list[i].compute_value_gradient(x[i], y[i])
            value_all = value_all + value
            grad_all = np.concatenate((grad_all, grad))
        return (value_all, grad_all)

    def compute_value_composite(self, x_slice, y_slice, additive_index):
        value = self.kernel_list[additive_index].compute_value(x_slice, y_slice)
        return value




# For test purpose
if __name__ == '__main__':
    with open('src\\Examples\\testadd_hyper2.py','r') as inf:
        HyperDicts = eval(inf.read())
    cur_bo = BayesOptimizationAdd(HyperDicts)
    xlist = []
    ylist = []
    for i in range(50):
        newX = cur_bo.sampler.sample()
        xlist.append(newX)
        tempX = {}
        for newDict in newX:
            tempX.update(newDict)
        newY = (tempX['x1'] + 0.3)**2 + (np.log(tempX['x2']) - 2)**2 + 3*(tempX['x3'] - 0.8)**2 \
        + tempX['x4']
        ylist.append(newY)
    t_start = time.time()
    xNext = cur_bo.suggestNext(xlist, ylist)
    t_end = time.time()
    print(xNext)
    print(t_end - t_start)


