# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import os, sys
import numpy as np
from cntk import DeviceDescriptor

TOLERANCE_ABSOLUTE = 1E-1  # TODO: Once set_fixed_random_seed(1) is honored, this must be tightened a lot.

from cntk.layers import *
from cntk.utils import *
from cntk.ops import splice

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, "..", "..", "..", "..", "Examples", "LanguageUnderstanding", "ATIS", "Python"))
sys.path.append("../LanguageUnderstanding/ATIS/Python")
from LanguageUnderstanding import data_dir, create_reader, create_model, train, emb_dim, hidden_dim, label_dim

def run_model_test(what, model, expected_train):
    print("--- {} ---".format(what))
    # train
    reader = create_reader(data_dir + "/atis.train.ctf", is_training=True)
    loss, metric = train(reader, model, max_epochs=1)
    print("-->", metric, loss)
    assert np.allclose([metric, loss], expected_train, atol=TOLERANCE_ABSOLUTE)

def create_test_model():
    # this selects additional nodes and alternative paths
    with default_options(enable_self_stabilization=True, use_peepholes=True):
        return Sequential([
            Stabilizer(),
            Embedding(emb_dim),
            BatchNormalization(),
            Recurrence(LSTM(hidden_dim, cell_shape=hidden_dim+50), go_backwards=True),
            BatchNormalization(map_rank=1),
            Dense(label_dim)
        ])

def with_lookahead():
    x = Placeholder()
    future_x = future_value(x)
    apply_x = splice (x, future_x)
    return apply_x

def BiRecurrence(fwd, bwd):
    F = Recurrence(fwd)
    G = Recurrence(fwd, go_backwards=True)
    x = Placeholder()
    apply_x = splice (F(x), G(x))
    return apply_x

def BNBiRecurrence(fwd, bwd, test_dual=True): # special version that calls one shared BN instance at two places, for testing BN param tying
    F = Recurrence(fwd)
    G = Recurrence(fwd, go_backwards=True)
    BN = BatchNormalization(normalization_time_constant=-1)
    x = Placeholder()
    # The following code applies the same BN function object twice.
    # When running whole-corpus estimation of means/vars, this must lead to the same estimate
    # although it is estimated on twice the amount of data (each sample is used twice).
    # Hence, this is the test that proves that the parameter sharing works.
    x1 = BN(x)
    x2 = BN(x) if test_dual else x1
    # In double precision with corpus aggregation, these lead to the same result.
    apply_x = splice (F(x1), G(x2))
    return apply_x

# TODO: the name is wrong
def test_language_understanding(device_id):
    from cntk.ops.tests.ops_test_utils import cntk_device
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
        #    run_model_test('change to intent classifier', Sequential([
        #        Embedding(emb_dim),
        #        with_lookahead(),
        #        BatchNormalization(),
        #        BiRecurrence(LSTM(hidden_dim)),
        #        BatchNormalization(),
        #        select_last,  # fails here with an axis problem
        #        Dense(label_dim)
        #    ]), [0.084, 0.407364])


        # replace lookahead by bidirectional model
        with default_options(initial_state=0.1):  # inject an option to mimic the BS version identically; remove some day
            run_model_test('replace lookahead by bidirectional model', Sequential([
                Embedding(emb_dim),
                BatchNormalization(),
                BiRecurrence(LSTM(hidden_dim), LSTM(hidden_dim)),
                BatchNormalization(),
                Dense(label_dim)
            ]), [0.0579573500457558, 0.3214986774820327])

        # replace lookahead by bidirectional model
        with default_options(initial_state=0.1):  # inject an option to mimic the BS version identically; remove some day
          #with default_options(dtype=np.float64):  # test this with double precision since single precision is too little for reproducable aggregation
          # ^^ This test requires to change the #if 1 in Functions.cpp PopulateNetworkInputs() to be changed to #if 0.
            run_model_test('replace lookahead by bidirectional model, with shared BN', Sequential([
                Embedding(emb_dim),
                BNBiRecurrence(LSTM(hidden_dim), LSTM(hidden_dim), test_dual=True),
                #BNBiRecurrence(LSTM(hidden_dim), LSTM(hidden_dim), test_dual=False),
                BatchNormalization(normalization_time_constant=-1),
                Dense(label_dim)
            ]), [0.0579573500457558, 0.3214986774820327])
            # values with normalization_time_constant=-1 and double precision:
            # [0.0583178503091983, 0.3199431143304898]
            """ with normalization_time_constant=-1:
             Minibatch[   1-   1]: loss = 5.945220 * 67, metric = 100.0% * 67
             Minibatch[   2-   2]: loss = 4.850601 * 63, metric = 79.4% * 63
             Minibatch[   3-   3]: loss = 3.816031 * 68, metric = 57.4% * 68
             Minibatch[   4-   4]: loss = 2.213172 * 70, metric = 41.4% * 70
             Minibatch[   5-   5]: loss = 2.615342 * 65, metric = 40.0% * 65
             Minibatch[   6-   6]: loss = 2.360896 * 62, metric = 25.8% * 62
             Minibatch[   7-   7]: loss = 1.452822 * 58, metric = 27.6% * 58
             Minibatch[   8-   8]: loss = 0.947210 * 70, metric = 10.0% * 70
             Minibatch[   9-   9]: loss = 0.595654 * 59, metric = 10.2% * 59
             Minibatch[  10-  10]: loss = 1.515479 * 64, metric = 23.4% * 64
             Minibatch[  11- 100]: loss = 0.686744 * 5654, metric = 10.4% * 5654
             Minibatch[ 101- 200]: loss = 0.289059 * 6329, metric = 5.8% * 6329
             Minibatch[ 201- 300]: loss = 0.218765 * 6259, metric = 4.7% * 6259
             Minibatch[ 301- 400]: loss = 0.182855 * 6229, metric = 3.5% * 6229
             Minibatch[ 401- 500]: loss = 0.156745 * 6289, metric = 3.4% * 6289
            Finished Epoch [1]: [Training] loss = 0.321413 * 36061, metric = 5.8% * 36061
            --> 0.057818696098277916 0.3214128415043278
             Minibatch[   1-   1]: loss = 0.000000 * 991, metric = 2.5% * 991
             Minibatch[   2-   2]: loss = 0.000000 * 1000, metric = 2.8% * 1000
             Minibatch[   3-   3]: loss = 0.000000 * 992, metric = 4.0% * 992
             Minibatch[   4-   4]: loss = 0.000000 * 989, metric = 3.0% * 989
             Minibatch[   5-   5]: loss = 0.000000 * 998, metric = 3.8% * 998
             Minibatch[   6-   6]: loss = 0.000000 * 995, metric = 1.5% * 995
             Minibatch[   7-   7]: loss = 0.000000 * 998, metric = 2.5% * 998
             Minibatch[   8-   8]: loss = 0.000000 * 992, metric = 1.6% * 992
             Minibatch[   9-   9]: loss = 0.000000 * 1000, metric = 1.6% * 1000
             Minibatch[  10-  10]: loss = 0.000000 * 996, metric = 7.9% * 996
            Finished Epoch [1]: [Evaluation] loss = 0.000000 * 10984, metric = 3.2% * 10984
            --> 0.03159140568099053 0.0
            """

        # BatchNorm test case for global-corpus aggregation
        with default_options(initial_state=0.1):  # inject an option to mimic the BS version identically; remove some day
            run_model_test('BatchNorm global-corpus aggregation', Sequential([
                Embedding(emb_dim),
                BatchNormalization(normalization_time_constant=-1),
                Recurrence(LSTM(hidden_dim), go_backwards=False),
                BatchNormalization(normalization_time_constant=-1),
                Dense(label_dim)
            ]), [0.05662627214996811, 0.2968516879905391])
            """
             Minibatch[   1-   1]: loss = 5.745576 * 67, metric = 100.0% * 67
             Minibatch[   2-   2]: loss = 4.684151 * 63, metric = 90.5% * 63
             Minibatch[   3-   3]: loss = 3.957423 * 68, metric = 63.2% * 68
             Minibatch[   4-   4]: loss = 2.286908 * 70, metric = 41.4% * 70
             Minibatch[   5-   5]: loss = 2.733978 * 65, metric = 38.5% * 65
             Minibatch[   6-   6]: loss = 2.189765 * 62, metric = 30.6% * 62
             Minibatch[   7-   7]: loss = 1.427890 * 58, metric = 25.9% * 58
             Minibatch[   8-   8]: loss = 1.501557 * 70, metric = 18.6% * 70
             Minibatch[   9-   9]: loss = 0.632599 * 59, metric = 13.6% * 59
             Minibatch[  10-  10]: loss = 1.516047 * 64, metric = 23.4% * 64
             Minibatch[  11- 100]: loss = 0.580329 * 5654, metric = 9.8% * 5654
             Minibatch[ 101- 200]: loss = 0.280317 * 6329, metric = 5.6% * 6329
             Minibatch[ 201- 300]: loss = 0.188372 * 6259, metric = 4.1% * 6259
             Minibatch[ 301- 400]: loss = 0.170403 * 6229, metric = 3.9% * 6229
             Minibatch[ 401- 500]: loss = 0.159605 * 6289, metric = 3.4% * 6289
            Finished Epoch [1]: [Training] loss = 0.296852 * 36061, metric = 5.7% * 36061
            --> 0.05662627214996811 0.2968516879905391
             Minibatch[   1-   1]: loss = 0.000000 * 991, metric = 1.8% * 991
             Minibatch[   2-   2]: loss = 0.000000 * 1000, metric = 3.4% * 1000
             Minibatch[   3-   3]: loss = 0.000000 * 992, metric = 3.9% * 992
             Minibatch[   4-   4]: loss = 0.000000 * 989, metric = 4.1% * 989
             Minibatch[   5-   5]: loss = 0.000000 * 998, metric = 4.0% * 998
             Minibatch[   6-   6]: loss = 0.000000 * 995, metric = 1.2% * 995
             Minibatch[   7-   7]: loss = 0.000000 * 998, metric = 2.8% * 998
             Minibatch[   8-   8]: loss = 0.000000 * 992, metric = 2.9% * 992
             Minibatch[   9-   9]: loss = 0.000000 * 1000, metric = 2.0% * 1000
             Minibatch[  10-  10]: loss = 0.000000 * 996, metric = 8.2% * 996
            Finished Epoch [1]: [Evaluation] loss = 0.000000 * 10984, metric = 3.5% * 10984
            --> 0.035050983248361256 0.0
            """


        # plus BatchNorm
        with default_options(initial_state=0.1):  # inject an option to mimic the BS version identically; remove some day
            run_model_test('plus BatchNorm', Sequential([
                Embedding(emb_dim),
                BatchNormalization(),
                Recurrence(LSTM(hidden_dim), go_backwards=False),
                BatchNormalization(),
                Dense(label_dim)
            ]), [0.05662627214996811, 0.2968516879905391])

        # plus lookahead
        with default_options(initial_state=0.1):  # inject an option to mimic the BS version identically; remove some day
            run_model_test('plus lookahead', Sequential([
                Embedding(emb_dim),
                with_lookahead(),
                BatchNormalization(),
                Recurrence(LSTM(hidden_dim), go_backwards=False),
                BatchNormalization(),
                Dense(label_dim)
            ]), [0.057901888466764646, 0.3044637752807047])

        # replace lookahead by bidirectional model
        with default_options(initial_state=0.1):  # inject an option to mimic the BS version identically; remove some day
            run_model_test('replace lookahead by bidirectional model', Sequential([
                Embedding(emb_dim),
                BatchNormalization(),
                BiRecurrence(LSTM(hidden_dim), LSTM(hidden_dim)),
                BatchNormalization(),
                Dense(label_dim)
            ]), [0.0579573500457558, 0.3214986774820327])

        # test of a config like in the example but with additions to test many code paths
        with default_options(enable_self_stabilization=True, use_peepholes=True):
                run_model_test('alternate paths', Sequential([
                Stabilizer(),
                Embedding(emb_dim),
                BatchNormalization(),
                Recurrence(LSTM(hidden_dim, cell_shape=hidden_dim+50), go_backwards=True),
                BatchNormalization(map_rank=1),
                    Dense(label_dim)
                ]), [0.08574360112032389, 0.41847621578367716])

    # test of the example itself
    # this emulates the main code in the PY file
    reader = create_reader(data_dir + "/atis.train.ctf", is_training=True)
    model = create_model()
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
    test_language_understanding(0)
