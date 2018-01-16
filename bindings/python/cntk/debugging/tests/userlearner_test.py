from __future__ import print_function

import numpy as np

import cntk as C
from cntk import Axis, NDArrayView
from cntk.logging import ProgressPrinter
from cntk.learners import UserLearner, sgd, learning_parameter_schedule
from cntk.layers import Dense, Sequential
import pytest


SEED = 1


def generate_random_data(sample_size, feature_dim, num_classes):
    # Create synthetic data using NumPy.
    Y = np.random.randint(size=(sample_size, 1), low=0, high=num_classes)

    # Make sure that the data is separable
    X = (np.random.randn(sample_size, feature_dim) + 3) * (Y + 1)
    X = X.astype(np.float32)
    # converting class 0 into the vector "1 0 0",
    # class 1 into vector "0 1 0", ...
    class_ind = [Y == class_number for class_number in range(num_classes)]
    Y = np.asarray(np.hstack(class_ind), dtype=np.float32)
    return X, Y


class MySgdNaive(UserLearner):

    def __init__(self, parameters, lr_schedule):
        super(MySgdNaive, self).__init__(parameters, lr_schedule)

    def update(self, gradient_values, training_sample_count, sweep_end):
        eta = self.learning_rate() / training_sample_count
        for p, g in gradient_values.items():
            new_p = p - eta * C.constant(g)
            p.set_value(new_p.eval(as_numpy=False).data)
        return True


class MySgdFast(UserLearner):

    def __init__(self, parameters, lr_schedule):
        super(MySgdFast, self).__init__(parameters, lr_schedule, as_numpy=False)

        self.new_p = {}
        self.grad_input = {}

        self.sample_count_input = C.input_variable(shape=(), name='count')

        lr = lr_schedule[0]  # assuming constant learning rate
        eta = lr / self.sample_count_input

        # we need one graph per parameter shape
        for param in parameters:
            p_shape = param.shape
            self.grad_input[p_shape] = C.input_variable(p_shape)
            self.new_p[p_shape] = param - eta * self.grad_input[p_shape]

    def update(self, gradient_values, training_sample_count, sweep_end):
        for p, g in gradient_values.items():
            new_p = self.new_p[p.shape]
            grad_input = self.grad_input[p.shape]

            data = {
                    self.sample_count_input: np.asarray(training_sample_count),
                    grad_input: g
                    }
            result = new_p.eval(data, as_numpy=False)
            shape = result.shape

            static_tensor = result.data.slice_view([0]*len(shape),
                                                   shape[1:])
            p.set_value(static_tensor)

        return True

ADDITIONAL_ARGUMENTS = [
    #(additional learning rate arguments (args), additional learner arguments (kwargs))
    (C.learning_rate_schedule, [C.learners.UnitType.minibatch], {'minibatch_size': 0}), #for backward compatible test
    (C.learning_parameter_schedule, [25], {'minibatch_size': 25}),  # test new API; 25 is the actually minibatch size
    (C.learning_parameter_schedule, [], {'minibatch_size': 0}),  # test new API
]

def ffnet(optimizer,  num_minibatches_to_train, learning_rate_func, lr_args, learner_kwargs):
    inputs = 2
    outputs = 2
    hidden_dimension = 50

    # input variables denoting the features and label data
    features = C.input_variable((inputs), np.float32)
    label = C.input_variable((outputs), np.float32)

    # Instantiate the feedforward classification model
    my_model = Sequential([
        Dense(hidden_dimension, activation=C.sigmoid,
              init=C.glorot_uniform(seed=SEED)),
        Dense(outputs, init=C.glorot_uniform(seed=SEED))])
    z = my_model(features)

    ce = C.cross_entropy_with_softmax(z, label)
    pe = C.classification_error(z, label)

    # Instantiate the trainer object to drive the model training
    lr= learning_rate_func(0.125, *lr_args)
    progress_printer = ProgressPrinter(0)
    learner = optimizer(z.parameters, lr) if optimizer != sgd else sgd(z.parameters, lr, **learner_kwargs)

    trainer = C.Trainer(z, (ce, pe), [learner], progress_printer)

    # Get minibatches of training data and perform model training
    minibatch_size = 25

    for i in range(num_minibatches_to_train):
        train_features, labels = generate_random_data(
            minibatch_size, inputs, outputs)
        # Specify the mapping of input variables in the model to actual
        # minibatch data to be trained with
        trainer.train_minibatch({features: train_features, label: labels})

    test_features, test_labels = generate_random_data(
        minibatch_size, inputs, outputs)
    avg_error = trainer.test_minibatch(
        {features: test_features, label: test_labels})
    print(' error rate on an unseen minibatch: {}'.format(avg_error))
    return z.parameters

@pytest.mark.parametrize("lr_func, lr_args, learner_kwargs", ADDITIONAL_ARGUMENTS)
def test_user_learner(lr_func, lr_args, learner_kwargs):
    num_minibatches_to_train = 10

    np.random.seed(SEED)
    # sort based on shape (this works because all parameters have different
    # shapes)
    p1 = sorted([p.value for p in ffnet(sgd, num_minibatches_to_train, lr_func, lr_args, learner_kwargs)], key=lambda x: x.shape)

    np.random.seed(SEED)
    p2 = sorted([p.value for p in ffnet(MySgdNaive, num_minibatches_to_train, lr_func, lr_args, learner_kwargs)], key=lambda x: x.shape)

    np.random.seed(SEED)
    p3 = sorted([p.value for p in ffnet(MySgdFast, num_minibatches_to_train, lr_func, lr_args, learner_kwargs)], key=lambda x: x.shape)

    for a, b, c in zip(p1, p2, p3):
        assert np.allclose(b, c)
        assert np.allclose(a, b)
        assert np.allclose(a, c)


if __name__ == '__main__':
    import timeit
    t1 = timeit.timeit("userlearner_test.ffnet(sgd, 1000)",
                       setup="from cntk.debugging.tests import userlearner_test; from cntk.learner import sgd",
                       number=10)
    print(t1)
    t2 = timeit.timeit("userlearner_test.ffnet(userlearner_test.MySgdNaive, 1000)",
                       setup="from cntk.debugging.tests import userlearner_test; from cntk.learner import sgd",
                       number=10)
    print(t2)
    t3 = timeit.timeit("userlearner_test.ffnet(userlearner_test.MySgdFast, 1000)",
                       setup="from cntk.debugging.tests import userlearner_test; from cntk.learner import sgd",
                       number=10)
    print(t3)
