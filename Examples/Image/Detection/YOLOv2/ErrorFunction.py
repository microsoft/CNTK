# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================


from cntk import user_function
from cntk.ops import *
import numpy as np
from PARAMETERS import *
from TrainUDFyolov2 import TrainFunction
from cntk_debug_single import DebugLayerSingle

def get_error(network, gtb_input, cntk_only=False):
    if cntk_only:
        assert False, "cntk_only loss function is no longer supported"
        err_f = ErrorFunction()
        return err_f.evaluate_network(network, gtb_input)

    else:
        ud_tf = TrainFunction(network, gtb_input)
        training_model = user_function(ud_tf)

        #err = TrainFunction.make_wh_sqrt(training_model.outputs[0]) - TrainFunction.make_wh_sqrt(network)
        targets = alias(training_model.outputs[0], 'TrainFunction_0')
        err_w= alias(training_model.outputs[1], 'TrainFunction_1')

        #network = user_function(DebugLayerSingle(network, debug_name='net_out', split_line=True))
        #targets = user_function(DebugLayerSingle(targets, debug_name='targets', split_line=True, print_grads=False))
        #err_w = user_function(DebugLayerSingle(err_w, debug_name='err_weights', split_line=True, print_grads=False))

        err = targets - network
        sq_err = err * err
        sc_err = sq_err * err_w # apply scales (lambda_coord, lambda_no_obj, zeros on not learned params)
        mse = reduce_sum(sc_err, axis=Axis.all_static_axes(), name="MeanSquaredError")
        return mse

def test_get_error():
    """
    Test for get_error()
    :return: Nothing
    """
    assert False, "Not implemented yet"


