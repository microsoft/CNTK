# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from .. import cntk_py 
from ..device import use_default_device
from cntk.internal import sanitize_var_map, sanitize_function, typemap
from ..io import MinibatchData

__doc__= '''\
An evaluator provides functionality to evaluate minibatches against the specified evaluation function.
'''

class Evaluator(cntk_py.Evaluator):
    '''
    Class for evaluation of minibatches against the specified evaluation function.

    Args:
       eval_function (:class:`~cntk.ops.functions.Function`): evaluation function.
       progress_writers (list): optionally, list of progress writers from :mod:`cntk.utils` to track
         training progress.
    '''

    def __init__(self, eval_function, progress_writers=None):
        if eval_function is not None:
            eval_function = sanitize_function(eval_function)

        if progress_writers is None:
            progress_writers = []
        elif not isinstance(progress_writers, list):
            progress_writers = [progress_writers]

        evaluator = cntk_py.create_evaluator(eval_function, progress_writers)
        # transplant into this class instance
        self.__dict__ = evaluator.__dict__

    def test_minibatch(self, arguments, device=None):
        '''
        Test the evaluation function on the specified batch of samples.

        Args:
            arguments: maps variables to their
             input data. The interpretation depends on the input type:

               * `dict`: keys are input variable or names, and values are the input data.
                 See :meth:`~cntk.ops.functions.Function.forward` for details on passing input data.

               * any other type: if node has an unique input, ``arguments`` is mapped to this input.
                 For nodes with more than one input, only `dict` is allowed.

             In both cases, every sample in the data will be interpreted
             as a new sequence. To mark samples as continuations of the
             previous sequence, specify ``arguments`` as `tuple`: the
             first element will be used as ``arguments``, and the second one will
             be used as a list of bools, denoting whether a sequence is a new
             one (`True`) or a continuation of the previous one (`False`).
             Data should be either NumPy arrays or a
             :class:`~cntk.io.MinibatchData` instance.
            device (:class:`~cntk.device.DeviceDescriptor`): the device descriptor that
             contains the type and id of the device on which the computation is
             to be performed.

        Note:
             See :meth:`~cntk.ops.functions.Function.forward` for examples on
             passing input data.

        Returns:
            `float`: the average evaluation criterion value per sample for the
              tested minibatch.
        '''
        if not device:
            device = use_default_device()

        arguments = sanitize_var_map(tuple(self.evaluation_function.arguments), arguments)
        return super(Evaluator, self).test_minibatch(arguments, device)
        
    @property
    @typemap
    def evaluation_function(self):
        '''
        The evaluation function that the evaluator is using.
        '''
        return super(Evaluator, self).evaluation_function()

    def summarize_test_progress(self):
        '''
        Updates the progress writers with the summary of test progress since start and resets the internal
        accumulators.
        '''
        return super(Evaluator, self).summarize_test_progress()
