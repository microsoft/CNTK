from cntk.learners import UserLearner
import cntk
import numpy as np
class PGDLearner(UserLearner):

    def __init__(self, parameters, lr_schedule, l1_regularization_weight=0.0, l2_regularization_weight=0.0):
        super(PGDLearner, self).__init__(parameters, lr_schedule)
        self.l1 = l1_regularization_weight
        self.l2 = l2_regularization_weight

    def update(self, gradient_values, training_sample_count, sweep_end):
        l1 = self.l1
        l2 = self.l2
        lr = self.learning_rate()
        
        for i in gradient_values:
            i.value -= gradient_values[i].to_ndarray()*lr
            i.value = np.sign(i.value) * np.maximum(np.abs(i.value) - lr*l1, 0.0) / (1.0 + l2*lr)

        return True