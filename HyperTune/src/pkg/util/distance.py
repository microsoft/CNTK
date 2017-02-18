#!/usr/bin/env python

# Copyright (c) Microsoft. All rights reserved.
# Licensed under custom Microsoft Research License
# See LICENSE.md file in the project root for full license information.

# define distance for different type of hyperparameters

import os
import sys
import numpy as np


class Distance:

    def __init__(self, domain):
        # Format: name : [type, range or list, distibution]
        self.Domain = domain

    def getdist_metric(self, x_value, y_value, hp_distri):
        if hp_distri == 'Uniform':
            return np.abs(x_value - y_value)
        elif hp_distri == 'LogUniform':
            return np.abs(np.log(x_value) - np.log(y_value))
        else:
            print("Unsupported Sampling Distribution")
            sys.exit(1)

    def getdist_binary(self, x_value, y_value, hp_distri):
        if hp_distri == 'Uniform':
            return float(x_value != y_value)
        else:
            print('Unsupported Sampling Distribution')
            sys.exit(1)

    def coordinate_dist(self, x, y):
        key_list = self.Domain.keys()
        result = []
        for key in key_list:
            [hp_type, hp_rl, hp_distri] = self.Domain[key]
            x_value = x[key]
            y_value = y[key]
            if hp_type == 'Continuous' or hp_type == 'Discrete' or hp_type == 'Enumerate':
                value = self.getdist_metric(x_value, y_value, hp_distri)
            elif hp_type == 'Categorical':
                value = self.getdist_binary(x_value, y_value, hp_distri)
            
            else:
                print('Unsupported Hyperparameter Type')
                sys.exit(1)
            result.append(value)
        return np.array(result)
