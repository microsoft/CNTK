from cntk.learners import *
import numpy as np

class Proximal_gd(UserLearner):
    
    """
    This is the implementation of "Proximal descent gradient" learner for deep learning:
    References:
    [1] https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf
    [2] https://www.cntk.ai/pythondocs/cntk.learners.html
    [3] https://en.wikipedia.org/wiki/Proximal_gradient_method
    """

    def __init__(self, parameters, lr_schedule, l1_regularization_weight = 0.0, l2_regularization_weight = 0.0):
        super(Proximal_gd, self).__init__(parameters, lr_schedule)
        ##### add more self parameters here
        self.l1_regularization_weight = l1_regularization_weight
        self.l2_regularization_weight = l2_regularization_weight
    
    def update(self, gradient_values, training_sample_count, sweep_end):
        l1_r = self.l1_regularization_weight
        l2_r = self.l2_regularization_weight
        lbda = self.learning_rate()/training_sample_count

        if l1_r < 0.0:
            l1_r = 0.0

        for p in gradient_values:
            prox = p.value - lbda * gradient_values[p].to_ndarray()
            p.value -= gradient_values[p].to_ndarray()*lbda
            p.value -= np.argmin( 1 / (2 * lbda) * np.sum(np.square(p.value - prox)) )
            p.value = np.sign(p.value) * np.maximum(np.abs(p.value) - lbda * l1_r, 0.0 ) / (1.0 + l2_r * lbda )

        return True
