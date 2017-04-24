# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 02:02:58 2017

@author: hjf
"""
from cntk.learners import UserLearner
import numpy as np

class Pro_grad(UserLearner):

    def __init__(self, parameters, lr_schedule, l1_regularization_weight=0.0, l2_regularization_weight=0.0):
        super(Pro_grad, self).__init__(parameters, lr_schedule)
        self.l1 = l1_regularization_weight
        self.l2 = l2_regularization_weight

    def update(self, gradient_values, training_sample_count, sweep_end):
        l1 = self.l1
        l2 = self.l2
        lr = self.learning_rate()
        
        if l1<0.0:
            l1 = 0.0
        for i in gradient_values:
            i.value -= gradient_values[i].to_ndarray()*lr
            i.value = np.sign(i.value) * np.maximum(np.abs(i.value) - lr*l1, 0.0) / (1.0 + l2*lr)

        return True