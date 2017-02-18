#!/usr/bin/env python

# Copyright (c) Microsoft. All rights reserved.
# Licensed under custom Microsoft Research License
# See LICENSE.md file in the project root for full license information.

# Recommender module provide three different recommender:
# Random, GP, additive GP

import os, sys
import numpy as np
from ..util.randsample import *
from ..bayesopt.bayesopt import *
from ..bayesopt.additive import *


class RecommenderBase:
    
    def __init__(self, hyper_param_spec, opt_model_spec):
        self.hyper_dicts = hyper_param_spec
        self.sampler = None

    def sample(self):
        return self.sampler.sample()

    def next_hyper_param(self, existing_results, exclude_list):
        # existingResult is a list of (hyper_param, value, anything)
        # return next point
        return {}

    def convert_to_single_dict(selfself, hyper_param):
        return hyper_param

class RandRecommender(RecommenderBase):

    def __init__(self, hyper_param_spec, opt_model_spec):
        self.hyper_dicts = hyper_param_spec
        self.sampler = RandomSampler(hyper_param_spec)

    def next_hyper_param(self, existing_results, exclude_list):
        return self.sampler.sample()

class GPRecommender(RecommenderBase):

    def __init__(self, hyper_param_spec, opt_model_spec):
        self.hyper_dicts = hyper_param_spec
        self.bayes_opt_method = BayesOptimization(hyper_param_spec, opt_model_spec)
        self.sampler = RandomSampler(hyper_param_spec)

    #existing_results is list of (hyper_param, best_error, best_epoch, exError, exVar, job_id, epoch_finished)
    def next_hyper_param(self, existing_results, exclude_list):
        xlist = [result[0] for result in existing_results]
        ylist = np.array([result[3] for result in existing_results])
        nlist = np.array([result[4] for result in existing_results])
        return self.bayes_opt_method.suggest_next(xlist, ylist, nlist, exclude_list)


class GPAddRecommender(GPRecommender):

    def __init__(self, hyper_param_specs, opt_model_spec):
        self.hyper_dicts = hyper_param_specs
        self.bayes_opt_method = BayesOptimizationAdd(hyper_param_specs, opt_model_spec)
        self.sampler = RandomSamplerAdd(hyper_param_specs)

    def convert_to_single_dict(selfself, hyper_params):
        hyper_param = {}
        for param in hyper_params:
            hyper_param.update(param)
        return hyper_param

