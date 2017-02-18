#!/usr/bin/env python

# Copyright (c) Microsoft. All rights reserved.
# Licensed under custom Microsoft Research License
# See LICENSE.md file in the project root for full license information.

# give a random sample of a set of hyperparameters

import os
import sys
import random
import numpy as np

class RandomSampler:
    def __init__(self, domain):
        # Format: name : [type, range or list, distibution]
        self.domain = domain
        self.dimension = len(domain)

    def getonesample_range(self, hp_range, hp_distri):
        if hp_distri == 'Uniform':
            if len(hp_range) == 2:
                return random.uniform(*hp_range)
            elif len(hp_range) == 3:  # in [low, high, resolution] format
                low = hp_range[0]
                high = hp_range[1]
                resolution = hp_range[2]
                v = random.uniform(low, high)
                return low + resolution * np.floor((v-low)/resolution)
            else:
                raise ValueError("Uniform distribution requires [low, high, (opt)resolution] as the range.")
        elif hp_distri == 'LogUniform':
            if len(hp_range) == 2:
                cur = random.uniform(*np.log(hp_range))
                return np.exp(cur)
            elif len(hp_range) == 3:  # in [low, high, resolution] format
                low = np.log(hp_range[0])
                high = np.log(hp_range[1])
                resolution = np.log(hp_range[2])
                v = random.uniform(low, high)
                return np.exp(low + resolution * np.floor((v - low) / resolution))
            else:
                raise ValueError("Uniform distribution requires [low, high, (opt)resolution] as the range.")
        else:
            raise ValueError("Unsupported Sampling Distribution.")

    def getonesample_list(self, hp_list, hp_distri):
        if hp_distri == 'Uniform':
            return random.choice(hp_list) 
        else:
            raise ValueError("Unsupported Sampling Distribution.")

    def sample(self):
        keylist = self.domain.keys()
        result = {}
        for key in keylist:
            [hp_type, hp_rl, hp_distri] = self.domain[key]
            if hp_type == 'Continuous':
                value = self.getonesample_range(hp_rl, hp_distri)
            elif hp_type == 'Discrete':
                value = self.getonesample_range(hp_rl, hp_distri)
                value = int(round(value))
            elif hp_type == 'Categorical' or hp_type == 'Enumerate':
                value = self.getonesample_list(hp_rl, hp_distri)
            else:
                raise ValueError("Unsupported Hyperparameter Type.")

            result[key] = value
        return result


class RandomSamplerAdd:
    def __init__(self, domains):
        self.Samplers = [RandomSampler(domain) for domain in domains]
        self.Dims = [len(domain) for domain in domains]

    def sample(self):
        separated_list = [Sampler.sample() for Sampler in self.Samplers]
        return separated_list

    def sample_composite(self, group_index):
        return self.Samplers[group_index].sample()