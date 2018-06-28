import time

import numpy as np
import cntk as C
from cntk.layers.typing import Sequence, Tensor

from cntk.debugging import start_profiler, stop_profiler, enable_profiler

from IPython.display import SVG, display

C.try_set_default_device(C.cpu())

input_dim_model = (1, 28)
input_dim = 28
num_output_classes = 10

train_file = 'Train-28xseq_cntk_text.txt'
test_file = 'Test-28xseq_cntk_text.txt'

def create_reader(path, is_training, input_dim, num_label_classes):
    ctf = C.io.CTFDeserializer(path, C.io.StreamDefs(
        labels=C.io.StreamDef(field='labels', shape=num_label_classes, is_sparse=False),
        features=C.io.StreamDef(field='features', shape=input_dim, is_sparse=False)
    ))

    return C.io.MinibatchSource(ctf, randomize=is_training, max_sweeps=C.io.INFINITELY_REPEAT if is_training else 1)

x = C.input_variable(**Sequence[Tensor[input_dim_model]])
y = C.input_variable(num_output_classes)

def create_model(features):
    with C.layers.default_options(init=C.glorot_uniform(), activation=C.relu):
        h = features
        h = C.layers.Convolution(filter_shape=(5, 5), num_filters=8, strides=(2,2), pad=True, sequential=True, name='first_conv')(h)
        print(h)
        h = C.layers.Convolution(filter_shape=(5, 5), num_filters=16, strides=(2,2), pad=True, sequential=True, name='second_conv')(h)
        print(h)
        h = C.sequence.unpack(h, 0.0, True)
        h = C.pad(h, pattern=[(0, 7), (0,0), (0,0)], mode=C.ops.CONSTANT_PAD, constant_value=0)
        h = C.slice(h, 0, 0, 7)
        r = C.layers.Dense(num_output_classes, activation=None, name='classify')(h)
        return r

z = create_model(x)

print('Output shape of the first convolution layer:', z.first_conv.shape)
print('Bias value of the last dense layer:', z.classify.b.value)

C.logging.log_number_of_parameters(z)

def create_criterion_function(model, labels):
    loss = C.cross_entropy_with_softmax(model, labels)
    errs = C.classification_error(model, labels)
    return loss, errs

def print_training_progress(trainer, mb, frequency, verbose=1):
    training_loss = "NA"
    eval_error = "NA"
    if mb%frequency == 0:
        training_loss = trainer.previous_minibatch_loss_average
        eval_error = trainer.previous_minibatch_evaluation_average
        if verbose:
            print("Minibatch: {0}, Loss: {1:.4f}, Error: {2:.2f}%".format(mb, training_loss, eval_error*100))
    return mb, training_loss, eval_error

def train_test(train_reader, test_reader, model_func, num_sweeps_to_train_with=10):
    model = model_func(x/255)

    loss, label_error = create_criterion_function(model, y)

    learning_rate = 0.2
    lr_schedule = C.learning_parameter_schedule(learning_rate)
    print(z.parameters)
    learner = C.sgd(z.parameters, lr_schedule)
    trainer = C.Trainer(z, (loss, label_error), [learner])

    # tweak the size here, as the previous static axis of dim 28 now also counts into minibatch_size. 
    minibatch_size = 64 * input_dim
    num_samples_per_sweep = 60000 * input_dim # 1600
    num_minibatches_to_train = (num_samples_per_sweep * num_sweeps_to_train_with) / minibatch_size

    input_map = {
        y : train_reader.streams.labels,
        x : train_reader.streams.features
    }

    training_progress_output_freq = 20

    print('Total minibatch to train: {0}'.format(int(num_minibatches_to_train)))

    start = time.time()

    #C.debugging.debug.set_node_timing(True)

    for i in range(0, int(num_minibatches_to_train)):
        data = train_reader.next_minibatch(minibatch_size, input_map=input_map)

        # check data here
        #print(data)
        #input('pause')

        trainer.train_minibatch(data)
        print_training_progress(trainer, i, training_progress_output_freq, verbose=1)
    
    print('Training took {:.1f} sec'.format(time.time() - start))
    #print(trainer.print_node_timing())
    input('pause')

    test_input_map = {
        y: test_reader.streams.labels,
        x: test_reader.streams.features
    }

    test_minibatch_size = 12 * (input_dim - 6)
    num_samples = 10000 * (input_dim - 6)
    num_minibatches_to_test = num_samples // test_minibatch_size

    test_result = 0.0

    print('Total minibatch to test: {0}'.format(int(num_minibatches_to_test)))

    for i in range(num_minibatches_to_test):
        data = test_reader.next_minibatch(test_minibatch_size, input_map=test_input_map)
        eval_error = trainer.test_minibatch(data)
        test_result = test_result + eval_error

    #    eval_error = trainer.test_minibatch(data)
    #    test_result = test_result + eval_error
    print('Average test error: {0:.2f}%'.format(test_result*100/num_minibatches_to_test))

def do_train_test():
    global z
    z = create_model(x)

    def display_model(model):
        svg = C.logging.graph.plot(model, 'tmp.svg')
        display(SVG(filename='tmp.svg'))
    display_model(z)

    input('pause')

    reader_train = create_reader(train_file, True, input_dim, num_output_classes)
    reader_test = create_reader(test_file, False, input_dim, num_output_classes)

    # profiling 
    #start_profiler(dir='seq_profiler')

    #enable_profiler()

    train_test(reader_train, reader_test, z)

    #stop_profiler()



do_train_test()

