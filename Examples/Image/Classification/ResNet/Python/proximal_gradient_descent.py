from cntk.learners import UserLearner

class proximal_gradient_descent(UserLearner):
    '''A proximal gradient descent learner. L1 and L2 regularization are present in the built-in
    learners but omitted here for simplicity.'''

    def __init__(self, parameters, lr_schedule):
        #just invoke the superclass constructor
        super(proximal_gradient_descent, self).__init__(parameters, lr_schedule)

    def update(self, gradient_values, training_sample_count, sweep_end):
        for var, value in gradient_values.items():
            var.value -= self.learning_rate() * value.to_ndarray()
            #might need some work here
        return True

