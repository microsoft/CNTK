# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import sys
from .. import cntk_py
from ..device import use_default_device
from ..utils import value_to_seq
from cntk.internal import sanitize_var_map, sanitize_function, typemap
from ..io import _py_dict_to_cntk_dict

__doc__ = '''\
A training session encapsulates a typical training loop and binds together a minibatch source that is used for training, a :doc:`trainer <cntk.train.trainer>` and an optional cross validation minibatch source. A training session takes care of consistent checkpointing and progress printing with specified frequencies. 
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
        callback (func (index, avarage_error, cv_num_samples, cv_num_minibatches)): Callback that will 
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
        var_to_stream (dict): mapping between input variables and input streams
        max_samples (int): maximum number of samples used for training
        progress_frequency (int): frequency in samples for aggregated progress printing
        checkpoint_config (:class:`CheckpointConfig`): checkpoint configuration
        cv_config (:class:`CrossValidationConfig`): cross validation configuration
        test_config (:class:`TestConfig`): test configuration
    '''
    def __init__(self, trainer, mb_source, mb_size,
                 var_to_stream, max_samples,
                 progress_frequency, 
                 checkpoint_config,
                 cv_config,
                 test_config):

        if trainer is None:
            raise ValueError("Trainer must not be None.")

        if mb_source is None:
            raise ValueError("Training minibatch source must not be None.")

        if var_to_stream is None or len(var_to_stream) == 0:
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
            var_to_stream, max_samples,  
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
    Create a minibatch size schedule

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
        schedule (integer or list): if integer, it this minibatch size will be used for the whole training.
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
def training_session(training_minibatch_source=None,        # deprecated, will be removed in the next version
                     trainer=None,                          
                     mb_size_schedule=None,                 # deprecated, will be removed in the next version
                     progress_printer=None,                 # deprecated, will be removed in the next version
                     model_inputs_to_mb_source_mapping={},  # deprecated, will be removed in the next version
                     checkpoint_filename=None,              # deprecated, will be removed in the next version
                     checkpoint_frequency=None,             # deprecated, will be removed in the next version
                     save_all_checkpoints=False,            # deprecated, will be removed in the next version
                     restore=True,                          # deprecated, will be removed in the next version
                     progress_frequency=None,               # deprecated, will be removed in the next version
                     cv_source=None,                        # deprecated, will be removed in the next version
                     cv_mb_size_schedule=None,              # deprecated, will be removed in the next version
                     cv_frequency=None,                     # deprecated, will be removed in the next version
                     max_training_samples=None,             # deprecated, will be removed in the next version
                     # new parameters with renaming
                     mb_source=None, 
                     mb_size=None,
                     var_to_stream=None, 
                     max_samples=None,
                     training_config=None,
                     progress_config=None,
                     checkpoint_config=None,
                     cv_config=None,
                     test_config=None):
    '''
    A factory function to create a training session object.

    Args: 
        training_minibatch_source (:class:`~cntk.io.MinibatchSource`): !DEPRECATED! use mb_source instead
        trainer (:class:`~cntk.train.trainer.Trainer`): trainer
        mb_size_schedule (:class:`~cntk.cntk_py.minibatch_size_schedule`): !DEPRECATED! use mb_size instead
        progress_printer (list): !DEPRECATED! list of progress writers from :mod:`cntk.utils`
        model_inputs_to_mb_source_mapping (dict): !DEPRECATED! use var_to_stream instead
        checkpoint_filename (str): !DEPRECATED! checkpoint file name.
        checkpoint_frequency (int): !DEPRECATED! checkpoint frequency in samples. If 0, no checkpointing takes place. 
          If ``sys.maxsize``, a single checkpoint is taken at the end of the training.
        save_all_checkpoints (bool): !DEPRECATED! saves all checkpoints, using ``checkpoint_filename`` as prefix and checkpoint index as a suffix.
        restore (bool): flag, indicating whether to restore from available checkpoint before the start of the training
        progress_frequency (int): frequency in samples for aggregated progress printing
        cv_source (:class:`~cntk.io.MinibatchSource`): !DEPRECATED! minibatch source used for cross validation
        cv_frequency (int): !DEPRECATED! frequency in samples for cross validation
        cv_mb_size_schedule (:class:`~cntk.cntk_py.minibatch_size_schedule`): !DEPRECATED! minibatch schedule for cross validation
          If ``sys.maxsize``, a single cross validation is performed at the end of training.
        max_training_samples (int): !DEPRECATED! use max_samples instead

        mb_source (:class:`~cntk.io.MinibatchSource`): minibatch source used for training
        mb_size (:class:`~cntk.cntk_py.minibatch_size_schedule`): minibatch schedule for training
        var_to_stream (dict): mapping between input variables and input streams
        max_samples (int): maximum number of samples used for training
        checkpoint_config (:class:`~CheckpointConfig`): checkpoint configuration
        cv_config (:class:`~CrossValidationConfig`): cross validation configuration

    Returns:
        Instance of :class:`~TrainingSession`
    '''
    if checkpoint_filename is not None or   \
       checkpoint_frequency is not None or  \
       save_all_checkpoints != False or     \
       restore != True or                   \
       cv_source is not None or             \
       cv_mb_size_schedule is not None or   \
       training_minibatch_source is not None or  \
       model_inputs_to_mb_source_mapping != {} or \
       max_training_samples is not None or  \
       cv_frequency is not None:
        import warnings
        warnings.warn('The provided parameters will be removed.'
                      ' All aspects of training session can be'
                      ' configured using config objects.')

    if mb_source is None:
        mb_source = training_minibatch_source

    if mb_size is None:
        mb_size = mb_size_schedule

    if var_to_stream is None:
        var_to_stream = model_inputs_to_mb_source_mapping

    if max_samples is None:
        max_samples = max_training_samples

    if checkpoint_config is None:
        checkpoint_config = CheckpointConfig(checkpoint_filename, checkpoint_frequency,
                                             restore, save_all_checkpoints)

    if cv_config is None:
       cv_config = CrossValidationConfig(
                cv_source, cv_mb_size_schedule, cv_frequency)

    if test_config is None:
       test_config = TestConfig(source=None)

    if progress_frequency != 0 and progress_printer is not None:
        cntk_py._add_progress_writers(trainer, [progress_printer])

    return TrainingSession(trainer, mb_source, mb_size, var_to_stream, max_samples,
                           progress_frequency, checkpoint_config, cv_config, test_config)
