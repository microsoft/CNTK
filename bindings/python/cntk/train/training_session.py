# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import sys
from enum import Enum, unique
from .. import cntk_py
from ..device import use_default_device
from cntk.internal import sanitize_var_map, sanitize_function, typemap, _as_tuple
import enum

class DataUnit(enum.IntEnum):

    '''
    Indicates that whether the processing steps in the training data is counted by samples, minibatch or epoch.
    '''

    sample = cntk_py.DataUnit_Sample
    '''
    Steps on data are counted by samples.
    '''

    minibatch = cntk_py.DataUnit_Minibatch
    '''
    Steps on data are counted by samples.
    '''

    sweep = cntk_py.DataUnit_Sweep
    '''
    Steps on data are counted by sweeps of epochs.
    '''


def _unpack_parameter_frequency(frequency):
    '''
    Return the a tuple (frequency, frequency_unit).
    The frequency_unit is either DataUnit_Sample, DataUnit_Minibatch, DataUnit_Sweep and default is DataUnit_Sample.
    '''
    if frequency is not None:
        if isinstance(frequency, int):
            #default to sample unit
            return frequency, DataUnit.sample
        elif isinstance(frequency, tuple) and isinstance(frequency[0], int) and isinstance(frequency[1], DataUnit):
            return frequency
        else:
            raise('Unsupported frequency specification: %s' % frequency)
    else:
        #default to sample unit
        return None, DataUnit.sample


__doc__ = '''\
A training session encapsulates a typical training loop and binds together a minibatch source that is used for training, a :class:`~cntk.train.trainer.Trainer` and an optional cross validation minibatch source. A training session takes care of consistent checkpointing and progress printing with specified frequencies.
'''
class CheckpointConfig(cntk_py.CheckpointConfig):
    '''
    A checkpoint configuration for the training session.

    Args:
        filename (str): checkpoint file name.
        frequency (int, tuple): checkpointing period (number samples between checkpoints).
          If `None`, no checkpointing takes place. 
          If ``sys.maxsize``, a single checkpoint is taken at the end of the training.
          If a tuple of (`frequency`, :class:`DataUnit`), the `frequency` is in terms of either `DataUnit.sample`, `DataUnit.minibatch` or `DataUnit.sweep`.
          See :class:`DataUnit` for more information on frequency data unit.
        restore (bool): flag, indicating whether to restore from available checkpoint before the start of the training
        preserve_all (bool): saves all checkpoints, using ``filename`` as prefix and checkpoint index as a suffix.
    '''
    def __init__(self, filename, frequency=None,
                 restore=True, preserve_all=False):
        '''Sets configuration of checkpointing behavior.

        Args:
            filename (str): checkpoint file name.
            frequency (int, tuple): checkpoint period (number samples between checkpoints). If 0, no checkpointing takes place.
              If ``sys.maxsize``, a single checkpoint is taken at the end of the training.
              If a tuple of (`frequency`, :class:`DataUnit`), the `frequency` is in terms of either `DataUnit.sample`, `DataUnit.minibatch` or `DataUnit.sweep`.
              See also:
                 :class:`DataUnit`
            restore (bool): flag, indicating whether to restore from available checkpoint before the start of the training
            preserve_all (bool): saves all checkpoints, using ``filename`` as prefix and checkpoint index as a suffix.

        Returns:
            Reconfigured self.
        '''
        frequency, frequency_unit = _unpack_parameter_frequency(frequency)
        if filename is None:
            if frequency is not None and frequency != 0:
                raise ValueError(
                    "Checkpoint frequency cannot be specified without checkpoint_filename")
            frequency = 0
            filename = ""

        if frequency is None:
            frequency = sys.maxsize

        super(CheckpointConfig, self).__init__(filename, frequency, frequency_unit,
                                               restore, preserve_all)

class CrossValidationConfig(cntk_py.CrossValidationConfig):
    '''
    A cross validation configuration for the training session.

    Args:
        minibatch_source (:class:`~cntk.io.MinibatchSource`): minibatch source used for cross validation
        frequency (int, tuple): frequency in samples for cross validation
         If None or ``sys.maxsize``, a single cross validation is performed at the end of training.
         If a tuple of (`frequency`, :class:`DataUnit`), the `frequency` is in terms of either `DataUnit.sample`, `DataUnit.minibatch` or `DataUnit.sweep`.
         See :class:`DataUnit` for more information on frequency data unit.
        minibatch_size(int or :class:`~cntk.cntk_py.minibatch_size_schedule`, defaults to 32): minibatch schedule for cross validation
        callback (func (index, average_error, cv_num_samples, cv_num_minibatches)): Callback that will
          be called with frequency which can implement custom cross validation logic,
          returns False if training should be stopped.
        max_samples (int, default None): number of samples to perform
          cross-validation on. If None, all samples are taken.
        model_inputs_to_streams (dict): mapping between input variables and input streams
          If None, the mapping provided to the training session constructor is used.
          Don't specify this if `minibatch_source` is a tuple of numpy/scipy arrays.
        criterion (:class:`~cntk.ops.functions.Function`): criterion function.
          Must be specified if `minibatch_source` is a tuple of numpy/scipy arrays.
        source (:class:`~cntk.io.MinibatchSource`): DEPRECATED, use minibatch_source instead
        mb_size(int or :class:`~cntk.cntk_py.minibatch_size_schedule`, defaults to 32): DEPRECATED, use minibatch_size instead
    '''
    def __init__(self, minibatch_source=None, frequency=None, minibatch_size=32,
            callback=None, max_samples=None, model_inputs_to_streams=None, criterion=None, source=None, mb_size=None):
        self.callback = callback
        frequency, frequency_unit = _unpack_parameter_frequency(frequency)

        if source is not None:
            self._warn_deprecated('"source" parameter is deprecated, please use "minibatch_source" instead')
            minibatch_source = source

        if mb_size is not None:
            self._warn_deprecated('"mb_size" parameter is deprecated, please use "minibatch_size" instead')
            minibatch_size = mb_size

        if minibatch_source is None and callback is None:
            if frequency is not None and frequency != 0:
                raise ValueError("Either minibatch_source of callback should be specified.")
            else:
                frequency = 0

        if frequency is None:
            frequency = sys.maxsize

        schedule = minibatch_size
        if isinstance(minibatch_size, int):
            schedule = minibatch_size_schedule(minibatch_size)

        if schedule is None:
            schedule = minibatch_size_schedule(1)

        if not isinstance(schedule, cntk_py.minibatch_size_schedule):
            raise ValueError('minibatch_size of type (%s) not supported. '
                             'it must be an output of minibatch_size_schedule() function'
                             % type(schedule))

        if max_samples is None:
            max_samples = sys.maxsize

        minibatch_source, model_inputs_to_streams = TrainingSession._sanitize_minibatch_source(minibatch_source, model_inputs_to_streams, criterion, infinitely_repeat=False)

        self._source_reference = minibatch_source # keep a Python-side strong reference so that SWIG finds the correct type upon callback (otherwise Python will crash)

        if model_inputs_to_streams is not None:
            super(CrossValidationConfig, self).__init__(
                minibatch_source, schedule, frequency, frequency_unit, max_samples, model_inputs_to_streams)
        else:
            super(CrossValidationConfig, self).__init__(
                minibatch_source, schedule, frequency, frequency_unit, max_samples)

    def _warn_deprecated(self, message):
        from warnings import warn
        warn('DEPRECATED: ' + message, DeprecationWarning, stacklevel=2)

class TestConfig(cntk_py.TestConfig):
    '''
    A test configuration for the training session.

    Args:
        minibatch_source (:class:`~cntk.io.MinibatchSource`): minibatch source used for cross validation
        minibatch_size(int or :class:`~cntk.cntk_py.minibatch_size_schedule`, defaults to 32): minibatch schedule for cross validation
        model_inputs_to_streams (dict): mapping between input variables and input streams
          If None, the mapping provided to the training session constructor is used.
          Don't specify this if `minibatch_source` is a tuple of numpy/scipy arrays.
        criterion (:class:`~cntk.ops.functions.Function`): criterion function.
          Must be specified if `minibatch_source` is a tuple of numpy/scipy arrays.
        source (:class:`~cntk.io.MinibatchSource`): DEPRECATED, use minibatch_source instead
        mb_size(int or :class:`~cntk.cntk_py.minibatch_size_schedule`, defaults to 32): DEPRECATED, use minibatch_size instead
    '''
    def __init__(self, minibatch_source=None, minibatch_size=32, model_inputs_to_streams=None, criterion=None, source=None, mb_size=None):
        if source is not None:
            self._warn_deprecated('"source" parameter is deprecated, please use "minibatch_source" instead')
            minibatch_source = source

        if mb_size is not None:
            self._warn_deprecated('"mb_size" parameter is deprecated, please use "minibatch_size" instead')
            minibatch_size = mb_size

        schedule = minibatch_size
        if isinstance(minibatch_size, int):
            schedule = minibatch_size_schedule(minibatch_size)

        if not isinstance(schedule, cntk_py.minibatch_size_schedule):
            raise ValueError('minibatch_size of type (%s) not supported. '
                             'it must be an int or the result of the minibatch_size_schedule() function'
                             % type(schedule))

        minibatch_source, model_inputs_to_streams = TrainingSession._sanitize_minibatch_source(minibatch_source, model_inputs_to_streams, criterion, infinitely_repeat=False)

        self._source_reference = minibatch_source # keep a Python-side strong reference so that SWIG finds the correct type upon callback (otherwise Python will crash)

        if model_inputs_to_streams is not None:
            super(TestConfig, self).__init__(minibatch_source, schedule, model_inputs_to_streams)
        else:
            super(TestConfig, self).__init__(minibatch_source, schedule)

    def _warn_deprecated(self, message):
        from warnings import warn
        warn('DEPRECATED: ' + message, DeprecationWarning, stacklevel=2)

class TrainingSession(cntk_py.TrainingSession):
    '''
    The instance of the class should be created by using :func:`~cntk.train.training_session.training_session` function.

    A training session trains a model using the specified ``trainer`` and configs.
    Different aspects of training such as data sources, checkpointing, cross validation, progress printing
    can be configured using the corresponding config classes.

    Args:
        trainer (:class:`~cntk.train.trainer.Trainer`): trainer
        mb_source (:class:`~cntk.io.MinibatchSource`): minibatch source used for training
        mb_size (:class:`~cntk.cntk_py.minibatch_size_schedule` or int): minibatch size schedule for training
        model_inputs_to_streams (dict): mapping between input variables and input streams
        max_samples (int): maximum number of samples used for training
        progress_frequency (int, tuple): the number of samples, minibatches, sweeps of epochs per which aggregated progress is printed
         If a tuple of (`frequency`, :class:`DataUnit`), the `frequency` is in terms of either `DataUnit.sample`, `DataUnit.minibatch` or `DataUnit.sweep`.
         See :class:`DataUnit` for more information on frequency data unit.
        checkpoint_config (:class:`CheckpointConfig`): checkpoint configuration
        cv_config (:class:`CrossValidationConfig`): cross validation configuration
        test_config (:class:`TestConfig`): test configuration
    '''
    def __init__(self, trainer, mb_source, mb_size,
                 model_inputs_to_streams, max_samples,
                 progress_frequency, 
                 checkpoint_config,
                 cv_config,
                 test_config):

        if trainer is None:
            raise ValueError("Trainer must not be None.")

        if mb_source is None:
            raise ValueError("Training minibatch source must not be None.")

        progress_frequency, progress_frequency_unit = _unpack_parameter_frequency(progress_frequency)

        mb_source, model_inputs_to_streams = TrainingSession._sanitize_minibatch_source(mb_source, model_inputs_to_streams, trainer.loss_function)

        if model_inputs_to_streams is None or len(model_inputs_to_streams) == 0:
            raise ValueError(
                "Mapping between input vars and streams should not be empty.")

        if max_samples is None:
            max_samples = sys.maxsize

        if progress_frequency is None:
            progress_frequency = sys.maxsize

        schedule = mb_size
        if isinstance(mb_size, int):
            schedule = minibatch_size_schedule(mb_size)

        if not isinstance(schedule, cntk_py.minibatch_size_schedule):
            raise ValueError('mb_size of type (%s) not supported. '
                             'it must be an output of minibatch_size_schedule() function'
                             % type(schedule))

        self.cv_callback = None
        if cv_config is not None:
            self.cv_callback = cv_config.callback

        self._callback_references = (mb_source, checkpoint_config, test_config) # keep a strong reference inside this object so that SWIG finds it

        super(TrainingSession, self).__init__(trainer, mb_source, schedule,
            model_inputs_to_streams, max_samples,  
            progress_frequency,
            progress_frequency_unit,
            checkpoint_config,
            cv_config,
            test_config)

    @staticmethod
    def _sanitize_minibatch_source(minibatch_source, model_inputs_to_streams, criterion, infinitely_repeat=True):
        '''
        Helper to wrap numpy/scipy data into a minibatch source.
        '''
        from ..io import MinibatchSource, UserMinibatchSource, MinibatchSourceFromData, INFINITELY_REPEAT
        if minibatch_source and not isinstance(minibatch_source, (MinibatchSource, UserMinibatchSource)): # UserMinibatchSource derives from cntk_py.SwigMinibatchSource, not MinibatchSource, for director purposes
            args = _as_tuple(minibatch_source) # the minibatch_source is a tuple of numpy or scipy arrays that we construct a source around
            # args can also be a tuple of numpy/scipy arrays; we will construct on the fly
            if criterion is None:
                raise ValueError("when passing data directly in place of a minibatch source, criterion must be given")
            params = criterion.arguments
            if len(params) != len(args):
                raise ValueError("to pass data directly in place of a minibatch source, pass a tuple of {} numpy or scipy arrays, in the order of the arguments of the criterion function. You passed {} value(s)"
                                 .format(len(params), len(args)))
            param_names = [param.name if param.name else "stream_%s" %  i for i, param in enumerate(params)] # names are for debugging...
            if len(params) != len(set(param_names)): # ...and for stream names and thus must be unique. If multiple inputs have the same names...
                param_names = ["stream_%s" % i for i, _ in enumerate(params)] # ...we fall back to generic names
            param_types = [param._type for param in params]
            max_samples = INFINITELY_REPEAT if infinitely_repeat else len(args[0]) # if not infinite then do one data pass
            minibatch_source = MinibatchSourceFromData({name: (input, type) for name, input, type in zip(param_names, args, param_types)}, max_samples=max_samples)
            if model_inputs_to_streams is not None:
                raise ValueError( "mapping must not be provided when data is passed directly")
            model_inputs_to_streams = {param: minibatch_source.streams[name] for param, name in zip(params, param_names)}
        return minibatch_source, model_inputs_to_streams

    @typemap
    def train(self, device=None):
        '''
        Perform training on a specified device.

        Args:
            device (:class:`~cntk.device.DeviceDescriptor`): the device descriptor containing
               the type and id of the device where training takes place.
        '''

        if not device:
            device = use_default_device()

        super(TrainingSession, self).train(device)

    def on_cross_validation_end(self, index, average_error, num_samples, num_minibatches):
        '''
        Callback that gets executed at the end of cross validation.

        Args:
            index (int): index of the current callback.
            average_error (float): average error for the cross validation
            num_samples (int): number of samples in cross validation
            num_minibatches (int): number of minibatch in cross validation

        Returns:
            True if training should continue, False otherwise.
        '''
        if self.cv_callback is not None:
            return self.cv_callback(index, average_error, num_samples, num_minibatches)
        else:
            return True

@typemap
def minibatch_size_schedule(schedule, epoch_size=1):
    '''
    Creates a minibatch size schedule.

    Examples:
        >>> # Use a fixed value 32 for all minibatches
        >>> s = minibatch_size_schedule(32)
        >>> s[0], s[1]
        (32, 32)

        >>> # Use minibatches of size 32 for the first 1000 samples, then 64 for the remaining ones
        >>> s = minibatch_size_schedule([32, 64], 1000)
        >>> s[0], s[1], s[1000], s[1001]
        (32, 32, 64, 64)

        >>> # Use 32 for the first 12 epochs, then 64 for the next 15,
        >>> # followed by 128 for the remaining ones, with a 100 samples in an epoch
        >>> s = minibatch_size_schedule([(12, 32), (15, 64), (1, 128)], 100)
        >>> s[0], s[1199], s[1200], s[2699], s[2700], s[5000]
        (32, 32, 64, 64, 128, 128)

    Args:
        schedule (int or list): if integer, this minibatch size will be used for the whole training.
         In case of list of integers, the elements are used as the values for ``epoch_size`` samples. 
         If list contains pair, the second element is used as a value for (``epoch_size`` x first element) samples
        epoch_size (int): number of samples as a scheduling unit.

    Returns:
        training parameter schedule
    '''
    if isinstance(schedule, int):
        if epoch_size != 1:
            raise ValueError('when providing the schedule as a number,'
                             ' epoch_size is ignored')
        return cntk_py.minibatch_size_schedule(schedule)
    from ..learners import _prepare_training_parameter_list
    if isinstance(schedule, list):
        schedule = _prepare_training_parameter_list(schedule)
        return cntk_py.minibatch_size_schedule(schedule, epoch_size)

    raise ValueError(
        'schedule must be either a float or a list, not %s' % type(schedule))

@typemap
def training_session(trainer,                          
                     mb_source, 
                     mb_size,
                     model_inputs_to_streams, 
                     progress_frequency=None,              
                     max_samples=None,
                     checkpoint_config=None,
                     cv_config=None,
                     test_config=None):
    '''
    A factory function to create a training session object.

    Args: 
        trainer (:class:`~cntk.train.trainer.Trainer`): trainer
        mb_source (:class:`~cntk.io.MinibatchSource`): minibatch source used for training
        mb_size (:class:`~cntk.cntk_py.minibatch_size_schedule`): minibatch schedule for training
        model_inputs_to_streams (dict): mapping between input variables and input streams
        progress_frequency (int, tuple): frequency in samples for aggregated progress printing
         If a tuple of (`frequency`, :class:`DataUnit`), the `frequency` is in terms of either `DataUnit.sample`, `DataUnit.minibatch` or `DataUnit.sweep`.
         See :class:`DataUnit` for more information on frequency data unit.
        max_samples (int): maximum number of samples used for training
        checkpoint_config (:class:`~CheckpointConfig`): checkpoint configuration
        cv_config (:class:`~CrossValidationConfig`): cross validation configuration
        test_config (:class:`~TestConfig`): test configuration

    Returns:
        Instance of :class:`~TrainingSession`
    '''
    if checkpoint_config is None:
        checkpoint_config = CheckpointConfig(filename=None)

    if cv_config is None:
       cv_config = CrossValidationConfig(None)

    if test_config is None:
       test_config = TestConfig(None)

    return TrainingSession(trainer, mb_source, mb_size, model_inputs_to_streams, max_samples,
                           progress_frequency, checkpoint_config, cv_config, test_config)
