#!/usr/bin/env python

# Copyright (c) Microsoft. All rights reserved.
# Licensed under custom Microsoft Research License
# See LICENSE.md file in the project root for full license information.

# Bayesian optimization and acquisition functions

import os
import sys
import numpy as np
from scipy.stats import norm
from .gauprocess import *
from ..util.randsample import *
import pdb
import time

class BayesOptimization:
    def __init__(self, domain, model_spec = {}):
        self.sampler = RandomSampler(domain)
        self.read_model_spec(model_spec)
        self.bayesian_opt_model = GaussianProcess(domain, self.kernel_type, self.base_noise)
        

    #Main interface
    def suggest_next(self, xlist, ylist, nlist, exclude_list):
        # return next point to be evaluated in BO
        self.data_size = len(xlist)
        n = len(xlist)
        if n != len(ylist):
            raise ValueError("Invalid Input in Bayes Optimization: lengths of xlist and ylist are not equal.")
        if n != len(nlist):
            raise ValueError("Invalid Input in Bayes Optimization: lengths of xlist and nlist are not equal.")
        if n < self.init_size:
            # Bayes model estimate unreliable in this case
            return self.sampler.sample()
        else:
            self.best_target = np.min(ylist)
            self.bayesian_opt_model.fit(xlist, ylist, nlist)
            x_best = self.select_best(exclude_list)

            return x_best


    #Auxiliary functions
    def select_best(self, exclude_list):
        #Use this after fit
        score_list = []
        for i in range(self.sample_size):
            x_new = self.sampler.sample()
            if not self.has_hp_evaluated(x_new, score_list):
                (mean, std) = self.bayesian_opt_model.predict(x_new)
                criterion = self.computeUCB(mean, std, self.sampler.dimension)
                # criterion = self.computeEI(mean, std)
                score_list.append((x_new, criterion))

        min_val = min([score[1] for score in score_list])
        ind_max = np.argmax([score[1] for score in score_list])
        (x_best, criterion_best) = score_list[ind_max]
        while x_best in exclude_list:
            score_list[ind_max][1] = min_val  #move to end
            ind_max = np.argmax([score[1] for score in score_list])
            (x_best, criterion_best) = score_list[ind_max]
        return x_best

    # existing_results is list of(hyper_param, criterion)
    def has_hp_evaluated(self, new_hp, existing_results):
        hp_evaluated = False
        for result in existing_results:
            if new_hp == result[0]:
                hp_evaluated = True
                break

        return hp_evaluated

    def read_model_spec(self, model_spec):
        #Default value
        # self.sample_size = 3000               # used in form grid for compute criteria
        # self.init_size = 6                    # least amount of samples required to use BO
        # self.lambda_const = 0.3
        # self.kernel_type = 'M52'
        # self.base_noise = 0.01

        if 'BOGridSize' in model_spec:
            self.sample_size = model_spec['BOGridSize']
        if 'BOInitSize' in model_spec:
            self.init_size = model_spec['BOInitSize']
        if 'BOLambdaConst' in model_spec:
            self.lambda_const = model_spec['BOLambdaConst']
        if 'BOKernelType' in model_spec:
            self.kernel_type = model_spec['BOKernelType']
        if 'BOBaseNoise' in model_spec:
            self.base_noise = model_spec['BOBaseNoise']
        return 0


    def set_sample_size(self, sample_size):
        self.sample_size = sample_size


    def computeEI(self, mean, std):
        # return expected improvement (find maximizer)
        gamma = (self.best_target - mean) / std
        inter_val = gamma*norm.cdf(gamma) + norm.pdf(gamma)
        return std*inter_val


    def computeUCB(self, mean, std, dim):
        # return upper confidence bound (negative to find maximizer)
        return - mean + self.lambda_const * np.sqrt(dim * np.log(self.data_size)) * std

