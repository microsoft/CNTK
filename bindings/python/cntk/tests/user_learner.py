from __future__ import print_function
import numpy as np
import cntk as C
from cntk.learner import UserLearner, sgd, learning_rate_schedule, UnitType
from cntk.utils import ProgressPrinter
from cntk.layers import Dense, Sequential

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
     
class MySgd(UserLearner):
    def __init__(self, parameters, learningRateSchedule):
        super(MySgd, self).__init__(parameters, learningRateSchedule)
    
    def update(self, gradient_values, training_sample_count, sweep_end):
        eta = self.learning_rate() / training_sample_count
        for p, g in gradient_values.items():
            newp = p - eta * C.constant(g)
            p.set_value(newp.eval(as_numpy=False).data())
        return True

def ffnet(optimizer):
    inputs = 2
    outputs = 2
    layers = 2
    hidden_dimension = 50

    # input variables denoting the features and label data
    features = C.input_variable((inputs), np.float32)
    label = C.input_variable((outputs), np.float32)

    # Instantiate the feedforward classification model
    my_model = Sequential ([
                    Dense(hidden_dimension, activation=C.sigmoid, init=C.glorot_uniform(seed=98052)),
                    Dense(outputs, init=C.glorot_uniform(seed=98052))])
    z = my_model(features)

    ce = C.cross_entropy_with_softmax(z, label)
    pe = C.classification_error(z, label)

    # Instantiate the trainer object to drive the model training
    lr_per_minibatch = learning_rate_schedule(0.125, UnitType.minibatch)
    trainer = C.Trainer(z, (ce, pe), [optimizer(z.parameters, lr_per_minibatch)])

    # Get minibatches of training data and perform model training
    minibatch_size = 25
    num_minibatches_to_train = 63

    pp = ProgressPrinter(0)
    for i in range(num_minibatches_to_train):
        train_features, labels = generate_random_data(minibatch_size, inputs, outputs)
        # Specify the mapping of input variables in the model to actual minibatch data to be trained with
        trainer.train_minibatch({features : train_features, label : labels})
        pp.update_with_trainer(trainer)

    last_avg_error = pp.avg_loss_since_start()

    test_features, test_labels = generate_random_data(minibatch_size, inputs, outputs)
    avg_error = trainer.test_minibatch({features : test_features, label : test_labels})
    print(' error rate on an unseen minibatch: {}'.format(avg_error))
    return z.parameters

def test_user_learner():
    np.random.seed(98052)
    # sort based on shape (this works because all parameters have different shapes)
    p1 = sorted([p.value for p in ffnet(sgd)],   key=lambda x:x.shape)
    np.random.seed(98052)
    p2 = sorted([p.value for p in ffnet(MySgd)], key=lambda x:x.shape)
    for a, b in zip(p1, p2):
        assert np.allclose(a, b)
