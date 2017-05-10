# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np

from cntk import sgd, Trainer, learning_rate_schedule, parameter, \
                 input, times, cross_entropy_with_softmax, \
                 classification_error, UnitType, combine
from cntk.debugging.debug import debug_model, _DebugNode


def _generate_random_data_sample(sample_size, feature_dim, num_classes):
    # Create synthetic data using NumPy.
    Y = np.random.randint(size=(sample_size, 1), low=0, high=num_classes)

    # Make sure that the data is separable
    X = (np.random.randn(sample_size, feature_dim) + 3) * (Y + 1)
    X = X.astype(np.float32)
    class_ind = [Y == class_number for class_number in range(num_classes)]
    Y = np.asarray(np.hstack(class_ind), dtype=np.float32)
    return X, Y


def _train(z, loss, eval_error,
           f_input, l_input,
           num_output_classes,
           steps):
    np.random.seed(0)

    input_dim = 2

    lr_schedule = learning_rate_schedule(0.5, UnitType.minibatch)

    learner = sgd(z.parameters, lr_schedule)
    trainer = Trainer(z, (loss, eval_error), [learner])

    minibatch_size = 10

    for i in range(steps):
        features, labels = _generate_random_data_sample(
            minibatch_size, input_dim, num_output_classes)

        trainer.train_minibatch({f_input: features, l_input: labels})


class InStream(object):
    def __init__(self, to_be_read):
        self.to_be_read = to_be_read

    def readline(self):
        if not self.to_be_read:
            return 'c'
        return self.to_be_read.pop(0)


class OutStream(object):
    def __init__(self):
        self.written = []

    def write(self, text):
        text = text.strip()
        if text not in ['',
                        _DebugNode.PROMPT_FORWARD.strip(),
                        _DebugNode.PROMPT_BACKWARD.strip()]:
            self.written.append(text)

    def flush(self):
        pass


def test_debug_1():
    input_dim = 2
    num_output_classes = 2

    f_input = input(input_dim, np.float32, needs_gradient=True, name='features')
    p = parameter(shape=(input_dim,), init=10, name='p')

    ins = InStream(['n', 'n', 'n'])
    outs = OutStream()

    z = times(f_input, p, name='z')
    z = debug_model(z, ins, outs)

    l_input = input(num_output_classes, np.float32, name='labels')
    loss = cross_entropy_with_softmax(z, l_input)
    eval_error = classification_error(z, l_input)

    _train(z, loss, eval_error,
           loss.find_by_name('features'),
           loss.find_by_name('labels'),
           num_output_classes, 1)

    # outs.written contains something like
    # =================================== forward  ===================================
    # Parameter('p', [], [2]) with uid 'Parameter4'
    # Input('features', [#, *], [2]) with uid 'Input3'
    # Times: Output('UserDefinedFunction12_Output_0', [#, *], [2]), Output('UserDefinedFunction15_Output_0', [], [2]) -> Output('z', [#, *], [2 x 2]) with uid 'Times21'
    # =================================== backward ===================================
    # Times: Output('UserDefinedFunction12_Output_0', [#, *], [2]), Output('UserDefinedFunction15_Output_0', [], [2]) -> Output('z', [#, *], [2 x 2]) with uid 'Times21'
    # Input('features', [#, *], [2]) with uid 'Input3'
    # Parameter('p', [], [2]) with uid 'Parameter4'   assert outs.written == out_stuff

    assert len(outs.written) == 8

    v_p = "Parameter('p', "
    v_i = "Input('features'"
    v_t = 'Times: '

    assert outs.written[0].startswith('=') and 'forward' in outs.written[0]
    line_1, line_2, line_3 = outs.written[1:4]

    assert outs.written[4].startswith('=') and 'backward' in outs.written[4]
    line_5, line_6, line_7 = outs.written[5:8]
    assert line_5.startswith(v_t)
    assert line_6.startswith(v_p) and line_7.startswith(v_i) or \
           line_6.startswith(v_i) and line_7.startswith(v_p)


def test_debug_multi_output():
    input_dim = 2
    num_output_classes = 2

    f_input = input(input_dim, np.float32, needs_gradient=True, name='features')
    p = parameter(shape=(input_dim,), init=10, name='p')

    comb = combine([f_input, p])

    ins = InStream(['n', 'n', 'n', 'n', 'n'])
    outs = OutStream()

    z = times(comb.outputs[0], comb.outputs[1], name='z')
    z = debug_model(z, ins, outs)

    l_input = input(num_output_classes, np.float32, name='labels')
    loss = cross_entropy_with_softmax(z, l_input)
    eval_error = classification_error(z, l_input)

    _train(z, loss, eval_error,
           loss.find_by_name('features'),
           loss.find_by_name('labels'),
           num_output_classes, 1)

    # outs.written contains something like
    # =================================== forward  ===================================
    # Parameter('p', [], [2]) with uid 'Parameter4'
    # Input('features', [#, *], [2]) with uid 'Input3'
    # Times: Output('UserDefinedFunction12_Output_0', [#, *], [2]), Output('UserDefinedFunction15_Output_0', [], [2]) -> Output('z', [#, *], [2 x 2]) with uid 'Times21'
    # =================================== backward ===================================
    # Times: Output('UserDefinedFunction12_Output_0', [#, *], [2]), Output('UserDefinedFunction15_Output_0', [], [2]) -> Output('z', [#, *], [2 x 2]) with uid 'Times21'
    # Input('features', [#, *], [2]) with uid 'Input3'
    # Parameter('p', [], [2]) with uid 'Parameter4'   assert outs.written == out_stuff

    assert len(outs.written) == 8

    v_p = "Parameter('p', "
    v_i = "Input('features'"
    v_t = 'Times: '

    assert outs.written[0].startswith('=') and 'forward' in outs.written[0]
    line_1, line_2, line_3 = outs.written[1:4]

    assert outs.written[4].startswith('=') and 'backward' in outs.written[4]
    line_5, line_6, line_7 = outs.written[5:8]
    assert line_5.startswith(v_t)
    assert line_6.startswith(v_p) and line_7.startswith(v_i) or \
           line_6.startswith(v_i) and line_7.startswith(v_p)

