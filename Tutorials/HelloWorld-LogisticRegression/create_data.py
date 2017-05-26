# -*- coding: utf-8 -*-
"""
Copyright (c) Microsoft. All rights reserved.
Licensed under the MIT license. See LICENSE file in the project root for full license information.

"""

import numpy as np
from sklearn.utils import shuffle

# number of dimensions
Dim = 2

# number of samples
N_train = 1000
N_test = 500

def generate(N, mean, cov, diff):   
    #import ipdb;ipdb.set_trace()
    num_classes = len(diff)
    samples_per_class = int(N/2)

    X0 = np.random.multivariate_normal(mean, cov, samples_per_class)
    Y0 = np.zeros(samples_per_class)
    
    for ci, d in enumerate(diff):
        X1 = np.random.multivariate_normal(mean+d, cov, samples_per_class)
        Y1 = (ci+1)*np.ones(samples_per_class)
    
        X0 = np.concatenate((X0,X1))
        Y0 = np.concatenate((Y0,Y1))

    X, Y = shuffle(X0, Y0)
    
    return X,Y

def create_data_files(num_classes, diff, train_filename, test_filename, regression):
    print("Outputting %s and %s"%(train_filename, test_filename))
    mean = np.random.randn(num_classes)
    cov = np.eye(num_classes)      
    
    for filename, N in [(train_filename, N_train), (test_filename, N_test)]:
        X, Y = generate(N, mean, cov, diff)
        
        # output in CNTK Text format
        with open(filename, "w") as dataset:
            num_labels = int((1 + np.amax(Y)))
            for i in range(N):
                dataset.write("|features ")
                for d in range(Dim):
                    dataset.write("%f " % X[i,d])
                if (regression): 
                    dataset.write("|labels %f\n" % Y[i])
                else:
                    labels = ['0'] * num_labels;
                    labels[int(Y[i])] = '1'
                    dataset.write("|labels %s\n" % " ".join(labels))

def main():
    # random seed (create the same data)
    np.random.seed(10)

    create_data_files(Dim, [3.0], "Train_cntk_text.txt", "Test_cntk_text.txt", True)
    create_data_files(Dim, [[3.0], [3.0, 0.0]], "Train-3Classes_cntk_text.txt", "Test-3Classes_cntk_text.txt", False)
    
if __name__ == '__main__':
    main()
