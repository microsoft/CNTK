import math
import numpy as np
from cntk.learners import UseLearner
class new_sgd(UserLearner):
    def __init__(self, parameters, lr_schedule, l1_regularization_weight=0.00001, l2_regularization_weight=0.001):
        super(mysgd, self).__init__(parameters, lr_schedule)
        self.l1 = l1_regularization_weight
        self.l2 = l2_regularization_weight
        self.count = 0

    def update(self, gradient_values, training_sample_count, sweep_end):
        l1 = self.l1
        l2 = self.l2

        lr0 = self.learning_rate()  
        lr=lr0*pow(1.001,-self.count/training_sample_count)
               
            for t in gradient_values:
                t.value -= gradient_values[t].to_ndarray()*lr
                t.value = np.sign(t.value) * np.maximum(np.abs(t.value) - lr*l1, 0.0) / (1.0 + l2*lr)
       
        self.count +=1
        return True
