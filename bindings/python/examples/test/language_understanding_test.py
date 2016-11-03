# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
from cntk import DeviceDescriptor

TOLERANCE_ABSOLUTE = 1E-1

from cntk.blocks import *
from cntk.layers import *
from cntk.models import *
from cntk.utils import *
from cntk.ops import splice
from examples.LanguageUnderstanding.LanguageUnderstanding import data_dir, create_reader, create_model_function, train, evaluate, emb_dim, hidden_dim, num_labels
from cntk.persist import load_model, save_model

def test_a_model(what, model, expected_train, expected_test=None):
    print("--- {} ---".format(what))
    # train
    reader = create_reader(data_dir + "/atis.train.ctf", is_training=True)
    loss, metric = train(reader, model, max_epochs=1)
    print("-->", metric, loss)
    assert np.allclose([metric, loss], expected_train, atol=TOLERANCE_ABSOLUTE)
    # save and load--test this for as many configs as possible
    path = data_dir + "/model.cmf"
    #save_model(model, path)
    #model = load_model(path)
    # test
    reader = create_reader(data_dir + "/atis.test.ctf", is_training=False)
    loss, metric = evaluate(reader, model)
    print("-->", metric, loss)
    if expected_test is not None:
        assert np.allclose(metric, expected_test, atol=TOLERANCE_ABSOLUTE)

def create_test_model():
    # this selects additional nodes and alternative paths
    with default_options(enable_self_stabilization=True, use_peepholes=True):
        return Sequential([
            Stabilizer(),
            Embedding(emb_dim),
            BatchNormalization(),
            Recurrence(LSTM(hidden_dim, cell_shape=hidden_dim+50), go_backwards=True),
            BatchNormalization(map_rank=1),
            Dense(num_labels)
        ])

def with_lookahead():
    x = Placeholder()
    future_x = future_value(x)
    apply_x = splice ([x, future_x])
    return apply_x

def BiRecurrence(fwd, bwd):
    F = Recurrence(fwd)
    G = Recurrence(fwd, go_backwards=True)
    x = Placeholder()
    apply_x = splice ([F(x), G(x)])
    return apply_x

def BNBiRecurrence(fwd, bwd): # special version that calls one shared BN instance at two places, for testing BN param tying
    F = Recurrence(fwd)
    G = Recurrence(fwd, go_backwards=True)
    BN = BatchNormalization(normalization_time_constant=-1)
    x = Placeholder()
    apply_x = splice ([F(BN(x)), G(BN(x))])
    return apply_x

# TODO: the name is wrong
def test_seq_classification_error(device_id):
    DeviceDescriptor.set_default_device(cntk_device(device_id))

    from _cntk_py import set_computation_network_trace_level, set_fixed_random_seed
    #set_computation_network_trace_level(1)
    set_fixed_random_seed(1) # to become invariant to initialization order, which is a valid change
    # BUGBUG: This ^^ currently seems to have no impact; the two BN models below should be identical in training

    if device_id >= 0: # BatchNormalization currently does not run on CPU
        # change to intent classifier   --moved up here since this fails, as repro
        # BUGBUG: Broken, need to pass new criterion to train().
        #with default_options(initial_state=0.1):  # inject an option to mimic the BS version identically; remove some day
        #    select_last = slice(Placeholder(), Axis.default_dynamic_axis(), -1, 0)
        #    # BUGBUG: Fails with "RuntimeError: The specified dynamic axis named defaultDynamicAxis does not match any of the dynamic axes of the operand"
        #    test_a_model('change to intent classifier', Sequential([
        #        Embedding(emb_dim),
        #        with_lookahead(),
        #        BatchNormalization(),
        #        BiRecurrence(LSTM(hidden_dim)),
        #        BatchNormalization(),
        #        select_last,  # fails here with an axis problem
        #        Dense(num_labels)
        #    ]), [0.084, 0.407364])


        # replace lookahead by bidirectional model
        with default_options(initial_state=0.1):  # inject an option to mimic the BS version identically; remove some day
            test_a_model('replace lookahead by bidirectional model, with shared BN', Sequential([
                Embedding(emb_dim),
                #BatchNormalization(),
                BNBiRecurrence(LSTM(hidden_dim), LSTM(hidden_dim)),
                BatchNormalization(normalization_time_constant=-1),
                Dense(num_labels)
            ]), [0.0579573500457558, 0.3214986774820327], 0.028495994173343045)

        # replace lookahead by bidirectional model
        with default_options(initial_state=0.1):  # inject an option to mimic the BS version identically; remove some day
            test_a_model('replace lookahead by bidirectional model', Sequential([
                Embedding(emb_dim),
                BatchNormalization(normalization_time_constant=-1),
                BiRecurrence(LSTM(hidden_dim), LSTM(hidden_dim)),
                BatchNormalization(normalization_time_constant=-1),
                Dense(num_labels)
            ]), [0.0579573500457558, 0.3214986774820327], 0.028495994173343045)

        # BatchNorm test case for global-corpus aggregation
        with default_options(initial_state=0.1):  # inject an option to mimic the BS version identically; remove some day
            test_a_model('BatchNorm global-corpus aggregation', Sequential([
                Embedding(emb_dim),
                BatchNormalization(normalization_time_constant=-1),
                Recurrence(LSTM(hidden_dim), go_backwards=False),
                BatchNormalization(normalization_time_constant=-1),
                Dense(num_labels)
            ]), [0.05662627214996811, 0.2968516879905391], 0.035050983248361256)


        # plus BatchNorm
        with default_options(initial_state=0.1):  # inject an option to mimic the BS version identically; remove some day
            test_a_model('plus BatchNorm', Sequential([
                Embedding(emb_dim),
                BatchNormalization(),
                Recurrence(LSTM(hidden_dim), go_backwards=False),
                BatchNormalization(),
                Dense(num_labels)
            ]), [0.05662627214996811, 0.2968516879905391])

        # plus lookahead
        with default_options(initial_state=0.1):  # inject an option to mimic the BS version identically; remove some day
            test_a_model('plus lookahead', Sequential([
                Embedding(emb_dim),
                with_lookahead(),
                BatchNormalization(),
                Recurrence(LSTM(hidden_dim), go_backwards=False),
                BatchNormalization(),
                Dense(num_labels)
            ]), [0.057901888466764646, 0.3044637752807047])

        # replace lookahead by bidirectional model
        with default_options(initial_state=0.1):  # inject an option to mimic the BS version identically; remove some day
            test_a_model('replace lookahead by bidirectional model', Sequential([
                Embedding(emb_dim),
                BatchNormalization(),
                BiRecurrence(LSTM(hidden_dim), LSTM(hidden_dim)),
                BatchNormalization(),
                Dense(num_labels)
            ]), [0.0579573500457558, 0.3214986774820327])

        # test of a config like in the example but with additions to test many code paths
    with default_options(enable_self_stabilization=True, use_peepholes=True):
            test_a_model('alternate paths', Sequential([
            Stabilizer(),
            Embedding(emb_dim),
            BatchNormalization(),
            Recurrence(LSTM(hidden_dim, cell_shape=hidden_dim+50), go_backwards=True),
            BatchNormalization(map_rank=1),
                Dense(num_labels)
            ]), [0.08574360112032389, 0.41847621578367716])

    # test of the example itself
    # this emulates the main code in the PY file
    reader = create_reader(data_dir + "/atis.train.ctf", is_training=True)
    model = create_model_function()
    loss_avg, evaluation_avg = train(reader, model, max_epochs=1)
    expected_avg = [0.15570838301766451, 0.7846451368305728]
    assert np.allclose([evaluation_avg, loss_avg], expected_avg, atol=TOLERANCE_ABSOLUTE)

    # test of a config like in the example but with additions to test many code paths
    if device_id >= 0: # BatchNormalization currently does not run on CPU
        reader = create_reader(data_dir + "/atis.train.ctf", is_training=True)
        model = create_test_model()
        loss_avg, evaluation_avg = train(reader, model, max_epochs=1)
        log_number_of_parameters(model, trace_level=1) ; print()
        expected_avg = [0.084, 0.407364]
        assert np.allclose([evaluation_avg, loss_avg], expected_avg, atol=TOLERANCE_ABSOLUTE)
        # example also saves and loads; we skip it here, so that we get a test case of no save/load
        # (we save/load in all cases above)

    # test
    #reader = create_reader(data_dir + "/atis.test.ctf", is_training=False)
    #evaluate(reader, model)
    # BUGBUG: fails eval with "RuntimeError: __v2libuid__BatchNormalization456__v2libname__BatchNormalization11: inference mode is used, but nothing has been trained."

if __name__=='__main__':
    test_seq_classification_error(0)
