from cntk import UserLearner
import cntk
import numpy as np


class MySgdFast(UserLearner):

    def __init__(self, parameters, lr_schedule,l2_regularization_weight = 0.0):
	super(MySgdFast, self).__init__(parameters, lr_schedule, as_numpy=False)

	self.new_p = {}
	self.grad_input = {}

	self.sample_count_input = cntk.input((), name='count')

	self.lr=cntk.input(()) # assuming constant learning rate
	eta = self.lr / self.sample_count_input
	self.l2_regularization_weight=l2_regularization_weight
	# we need one graph per parameter shape
	for param in parameters:
	    p_shape = param.shape
	    self.grad_input[p_shape] = cntk.input(p_shape)
	    self.new_p[p_shape] =param*(1.0-self.l2_regularization_weight) -eta* self.grad_input[p_shape] /cntk.abs(
	                                                                                                           self.grad_input[p_shape])

    def update(self, gradient_values, training_sample_count, sweep_end):
	for p, g in gradient_values.items():
	    new_p = self.new_p[p.shape]
	    grad_input = self.grad_input[p.shape]

	    data = {
	        self.lr:np.asarray(self.learning_rate(),dtype=np.float32),
	        self.sample_count_input: np.asarray(training_sample_count,dtype=np.float32),
	        grad_input: g
	    }
	    result = new_p.eval(data, as_numpy=False)
	    shape = result.shape

	    # result has the shape of a complete minibatch, but contains
	    # only one tensor, which we want to write to p. This means, we
	    # have to slice off the leading dynamic axes.
	    static_tensor = result.data.slice_view([0]*len(shape),
	                                           shape[1:])
	    p.set_value(static_tensor)

	return True

