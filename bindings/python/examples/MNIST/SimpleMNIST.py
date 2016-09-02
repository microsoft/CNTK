import numpy as np
import sys
import os
from cntk import learning_rates_per_sample, DeviceDescriptor, Trainer, sgdlearner
from cntk.ops import variable, cross_entropy_with_softmax, combine, classification_error, sigmoid, element_times, constant
from cntk.utils import create_minibatch_source, get_train_loss, get_train_eval_criterion, cntk_device
from cntk.tests.test_utils import TOLERANCE_ABSOLUTE
from examples.common.nn import fully_connected_classifier_net
from examples.common.mb import create_text_mb_source

def simple_mnist():
    input_dim = 784
    num_output_classes = 10
    num_hidden_layers = 1
    hidden_layers_dim = 200
    epoch_size = sys.maxsize
    minibatch_size = 32
    num_samples_per_sweep = 60000
    num_sweeps_to_train_with = 1
    num_minibatches_to_train = (num_samples_per_sweep * num_sweeps_to_train_with) / minibatch_size
    lr = learning_rates_per_sample(0.003125)
    input = variable(input_dim, np.float32, needs_gradient=False, name="features")
    scaled_input = element_times(constant((), 0.00390625), input)

    label = variable(num_output_classes, np.float32, needs_gradient=False, name="labels")

    dev = -1
    cntk_dev = cntk_device(dev)
    netout = fully_connected_classifier_net(scaled_input, num_output_classes, hidden_layers_dim, num_hidden_layers, dev, sigmoid)

    ce = cross_entropy_with_softmax(netout, label)
    pe = classification_error(netout, label)

    #TODO: add save and load module code
    ffnet = combine([ce.owner, pe.owner, netout.owner], "classifier_model")

    rel_path = r"../../../../Examples/Image/MNIST/Data/Train-28x28_cntk_text.txt"
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), rel_path)
    cm = create_text_mb_source(path, input_dim, num_output_classes, epoch_size)

    stream_infos = cm.stream_infos()

    for si in stream_infos:
        if si.m_name == 'features':
            features_si = si
        elif si.m_name == 'labels':
            labels_si = si

    trainer = Trainer(netout.owner, ce.owner, pe.owner, [sgdlearner(netout.owner.parameters(), lr)])

    for i in range(0,int(num_minibatches_to_train)):
        mb=cm.get_next_minibatch(minibatch_size, cntk_dev)

        arguments = dict()
        arguments[input] = mb[features_si].m_data
        arguments[label] = mb[labels_si].m_data

        trainer.train_minibatch(arguments, cntk_dev)
        freq = 20
        if i % freq == 0:
            training_loss = get_train_loss(trainer)
            eval_crit = get_train_eval_criterion(trainer)
            print ("Minibatch: {}, Train Loss: {}, Train Evaluation Criterion: {}".format(i, training_loss, eval_crit))

if __name__=='__main__':
    simple_mnist()