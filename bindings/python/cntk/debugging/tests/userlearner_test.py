from __future__ import print_function

import numpy as np

import cntk as C
from cntk import input, Axis, NDArrayView
from cntk.logging import ProgressPrinter
from cntk.learners import UserLearner, sgd, learning_rate_schedule, UnitType
from cntk.layers import Dense, Sequential


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

        self.sample_count_input = input(shape=(), name='count')

        lr = lr_schedule[0]  # assuming constant learning rate
        eta = lr / self.sample_count_input

        # we need one graph per parameter shape
        for param in parameters:
            p_shape = param.shape
            self.grad_input[p_shape] = input(p_shape)
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


def ffnet(optimizer, num_minibatches_to_train):
    inputs = 2
    outputs = 2
    hidden_dimension = 50

    # input variables denoting the features and label data
    features = C.input((inputs), np.float32)
    label = C.input((outputs), np.float32)

    # Instantiate the feedforward classification model
    my_model = Sequential([
        Dense(hidden_dimension, activation=C.sigmoid,
              init=C.glorot_uniform(seed=SEED)),
        Dense(outputs, init=C.glorot_uniform(seed=SEED))])
    z = my_model(features)

    ce = C.cross_entropy_with_softmax(z, label)
    pe = C.classification_error(z, label)

    # Instantiate the trainer object to drive the model training
    lr_per_minibatch = learning_rate_schedule(0.125, UnitType.minibatch)
    progress_printer = ProgressPrinter(0)
    trainer = C.Trainer(z, (ce, pe), [optimizer(
        z.parameters, lr_per_minibatch)], progress_printer)

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


def test_user_learner():
    num_minibatches_to_train = 10

    np.random.seed(SEED)
    # sort based on shape (this works because all parameters have different
    # shapes)
    p1 = sorted([p.value for p in ffnet(sgd, num_minibatches_to_train)], key=lambda x: x.shape)

    np.random.seed(SEED)
    p2 = sorted([p.value for p in ffnet(MySgdNaive, num_minibatches_to_train)], key=lambda x: x.shape)

    np.random.seed(SEED)
    p3 = sorted([p.value for p in ffnet(MySgdFast, num_minibatches_to_train)], key=lambda x: x.shape)

    for a, b, c in zip(p1, p2, p3):
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
