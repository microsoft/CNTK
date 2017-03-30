# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import sys
from .. import cntk_py
from ..device import use_default_device
from cntk.internal import sanitize_var_map, sanitize_function, typemap
from ..io import _py_dict_to_cntk_dict

__doc__ = '''\
A training session encapsulates a typical training loop and binds together a minibatch source that is used for training, a :class:`~cntk.train.trainer.Trainer` and an optional cross validation minibatch source. A training session takes care of consistent checkpointing and progress printing with specified frequencies.
'''
class CheckpointConfig(cntk_py.CheckpointConfig):
    '''
    A checkpoint configuration for the training session.

    Args:
        filename (str): checkpoint file name.
        frequency (int): checkpoint frequency in samples. If 0, no checkpointing takes place. 
          If ``sys.maxsize``, a single checkpoint is taken at the end of the training.
        preserve_all (bool): saves all checkpoints, using ``filename`` as prefix and checkpoint index as a suffix.
        restore (bool): flag, indicating whether to restore from available checkpoint before the start of the training
    '''
    def __init__(self, filename, frequency=None,
                 restore=True, preserve_all=False):
        '''Sets configuration of checkpointing behavior.

        Args:
            filename (str): checkpoint file name.
            frequency (int): checkpoint frequency in samples. If 0, no checkpointing takes place. 
              If ``sys.maxsize``, a single checkpoint is taken at the end of the training.
            preserve_all (bool): saves all checkpoints, using ``filename`` as prefix and checkpoint index as a suffix.
            restore (bool): flag, indicating whether to restore from available checkpoint before the start of the training

        Returns:
            Reconfigured self.
        '''
        if filename is None:
            if frequency is not None and frequency != 0:
                raise ValueError(
                    "Checkpoint frequency cannot be specified without checkpoint_filename")
            frequency = 0
            filename = ""

        if frequency is None:
            frequency = sys.maxsize

        super(CheckpointConfig, self).__init__(filename, frequency,
                                               restore, preserve_all)

class CrossValidationConfig(cntk_py.CrossValidationConfig):
    '''
    A cross validation configuration for the training session.

    Args:
        source (:class:`~cntk.io.MinibatchSource`): minibatch source used for cross validation
        frequency (int): frequency in samples for cross validation
          If ``sys.maxsize``, a single cross validation is performed at the end of training.
        schedule (:class:`~cntk.cntk_py.minibatch_size_schedule`): minibatch schedule for cross validation
        callback (func (index, average_error, cv_num_samples, cv_num_minibatches)): Callback that will
          be called with frequency which can implement custom cross validation logic,
          returns False if training should be stopped.
    '''
    def __init__(self, source=None, mb_size=None, frequency=None, callback=None):
        self.callback = callback

        if source is None and callback is None:
            if frequency is not None and frequency != 0:
                raise ValueError("Either source of callback should be specified.")
            else:
                frequency = 0

        if frequency is None:
            frequency = sys.maxsize

        schedule = mb_size
        if isinstance(mb_size, int):
            schedule = minibatch_size_schedule(mb_size)

        if schedule is None:
            schedule = minibatch_size_schedule(1)

        if not isinstance(schedule, cntk_py.minibatch_size_schedule):
            raise ValueError('mb_size of type (%s) not supported. '
                             'it must be an output of minibatch_size_schedule() function'
                             % type(schedule))

        super(CrossValidationConfig, self).__init__(
            source, schedule, frequency)

class TestConfig(cntk_py.TestConfig):
    '''
    A test configuration for the training session.

    Args:
        source (:class:`~cntk.io.MinibatchSource`): minibatch source used for testing
        schedule (:class:`~cntk.cntk_py.minibatch_size_schedule`): minibatch schedule for testing
    '''
    def __init__(self, source, mb_size=None):
        schedule = mb_size
        if isinstance(mb_size, int):
            schedule = minibatch_size_schedule(mb_size)

        if schedule is None:
            schedule = minibatch_size_schedule(1)

        if not isinstance(schedule, cntk_py.minibatch_size_schedule):
            raise ValueError('mb_size of type (%s) not supported. '
                             'it must be an output of minibatch_size_schedule() function'
                             % type(schedule))

        super(TestConfig, self).__init__(source, schedule)

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
        progress_frequency (int): frequency in samples for aggregated progress printing
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

        super(TrainingSession, self).__init__(trainer, mb_source, schedule,
            model_inputs_to_streams, max_samples,  
            progress_frequency, 
            checkpoint_config,
            cv_config,
            test_config)

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

    if isinstance(schedule, list):
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
        progress_frequency (int): frequency in samples for aggregated progress printing
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
       cv_config = CrossValidationConfig(source=None)

    if test_config is None:
       test_config = TestConfig(source=None)

    return TrainingSession(trainer, mb_source, mb_size, model_inputs_to_streams, max_samples,
                           progress_frequency, checkpoint_config, cv_config, test_config)
