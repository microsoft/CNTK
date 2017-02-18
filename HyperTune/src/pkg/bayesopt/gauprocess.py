#!/usr/bin/env python

# Copyright (c) Microsoft. All rights reserved.
# Licensed under custom Microsoft Research License
# See LICENSE.md file in the project root for full license information.

# Gaussian process
# learn the hyperparameters of GP

import os
import sys
import numpy as np
import scipy.optimize as spopt
from .kernel import *
from scipy.stats import norm
import pdb


class GaussianProcess:


    def __init__(self, domain, kernel_type, base_noise):
        self.dimension = len(domain)
        self.kernel = Kernel(domain, kernel_type)
        self.base_noise = base_noise
        self.initialize()


    #Main interface
    def fit(self, xlist, ylist, nlist):
        self.xlist = xlist
        self.ylist = np.array(ylist)
        self.nlist = np.array(nlist)
        self.data_size = len(xlist)

        #TODO: multiple initialization and rerun
        x0 = self.feature_scale                              #initial guess

        # xresult = self.grad_descent(x0)
        # pdb.set_trace()
        # print xresult
        # self.check_grad(xresult)
        # pdb.set_trace()

        # res = spopt.minimize(self.compute_hyper_value_gradient, x0, method='CG', \
        #     jac=True, options={'gtol': 1e-3, 'disp': True})
        res = spopt.minimize(self.compute_hyper_value_gradient, x0, method='L-BFGS-B', \
                             jac=True, bounds = self.feature_scale_range, options={'gtol': 1e-3, 'disp': False})
        
        # forming kernel matrix
        self.feature_scale = res.x
        self.kernel.set_scale(self.feature_scale)
        self.compute_kernel_matrix_mean_var()

        # pdb.set_trace()


    #returns mean and std of the predicted value at x
    def predict(self, x):
        # use fit before predict
        self_kernel_value = self.kernel.compute_value(x, x)
        mutual_kernel_value = np.array([self.kernel.compute_value(xp, x) for xp in self.xlist])
        inv_kernel_matrix = self.inverse_kernel_matrix.dot(mutual_kernel_value)

        mean = self.function_mean + inv_kernel_matrix.dot(self.y_minus_mean)
        var = self.kernel_amplifier * (self_kernel_value - inv_kernel_matrix.dot(mutual_kernel_value))
        if var < 0:
            var = 0
        std = np.sqrt(var)
        return (mean, std)


    #Auxiliary functions
    def initialize(self):
        self.function_mean = 0
        self.kernel_amplifier = 1
        self.feature_scale = np.zeros(self.dimension)
        self.feature_scale_range = [(-10, 10)] * self.dimension



    def compute_kernel_matrix_mean_var(self):
        # forming kernel matrix
        kernel_matrix = np.empty([self.data_size, self.data_size])
        feature_scaled_gradient_tensor = np.empty([self.dimension, self.data_size, self.data_size])
        for i in range(self.data_size):
            for j in range(i+1):
                x = self.xlist[i]
                xp = self.xlist[j]
                (kernel_value, kernel_grad) = self.kernel.compute_value_gradient(x, xp)
                kernel_matrix[i, j] = kernel_value
                kernel_matrix[j, i] = kernel_value
                feature_scaled_gradient_tensor[:, i, j] = kernel_grad
                feature_scaled_gradient_tensor[:, j, i] = kernel_grad
        self.feature_scaled_gradient = feature_scaled_gradient_tensor
        
        # adding regularization and inverse
        # self.obNoise = np.linalg.norm(kernel_matrix) / 1e5
        base_noise_matrix = (self.base_noise ** 2) * np.identity(self.data_size)
        noise_matrix = base_noise_matrix + np.diag(self.nlist)

        self.y_kernel_matrix = kernel_matrix + noise_matrix
        self.inverse_kernel_matrix = np.linalg.inv(self.y_kernel_matrix)

        # calculate mean and kerAmp of the function
        temp_vec = self.inverse_kernel_matrix.dot(np.ones(self.data_size))
        self.function_mean = temp_vec.dot(self.ylist) / temp_vec.dot(np.ones(self.data_size))
        self.y_minus_mean = self.ylist - self.function_mean * np.ones(self.data_size)
        self.kernel_amplifier = self.inverse_kernel_matrix.dot(self.y_minus_mean).dot(self.y_minus_mean) / self.data_size



    def compute_hyper_value_gradient(self, feature_scale):
        # feature_scale structure:
        #     [0:]: feature scaling (feature_scale)
        #     [-1]: observation noise (obNoise)???
        self.kernel.set_scale(feature_scale)

        # forming kernel matrix and compute mean and var
        self.compute_kernel_matrix_mean_var()

        # compute marginal likelihood
        inverse_kernel_y = self.inverse_kernel_matrix.dot(self.y_minus_mean)
        (sign, logdet) = np.linalg.slogdet(self.y_kernel_matrix)
        neg_log_likelihood = 0.5 * (inverse_kernel_y.dot(self.y_minus_mean) / self.kernel_amplifier + logdet)  \
            + 0.5 * self.data_size * np.log(self.kernel_amplifier)

        # compute gradient
        grad_aux_matrix = np.outer(inverse_kernel_y, inverse_kernel_y) / self.kernel_amplifier - self.inverse_kernel_matrix
        # obNoise_grad = obNoise * np.trace(grad_aux_matrix)
        feature_scaled_grad = -0.5 * np.tensordot(self.feature_scaled_gradient, grad_aux_matrix)

        # concate_grad = np.array([funMean_grad, obNoise_grad, kerAmp_grad])
        # hyperGrad = np.concatenate([concate_grad, feature_scaled_grad])

        return (neg_log_likelihood, feature_scaled_grad)



    #Test functions
    def check_grad(self, x):
        # for testing purpose
        (val, grad) = self.compute_hyper_value_gradient(x)
        print(grad)
        for i in range(len(x)):
            eps = 1e-5
            xp = x.copy()
            xp[i] = xp[i] + eps
            (valp, gradp) = self.compute_hyper_value_gradient(xp)
            pgrad = (valp-val) / eps
            print(pgrad)

    
    def grad_descent(self, x0):
        x = x0
        for i in range(20):
            (val, grad) = self.compute_hyper_value_gradient(x)
            print(val, np.linalg.norm(grad))
            x = x - 0.0001*grad
        return x
        






# For test purpose
if __name__ == '__main__':
    with open('src\\Examples\\Simple_hyper.py','r') as inf:
        HyperDict = eval(inf.read())
    cur_GP = GaussianProcess(HyperDict)
    cur_GP.check_grad
