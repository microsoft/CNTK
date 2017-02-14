# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import sys
from . import cntk_py
from .device import use_default_device
from .utils import sanitize_var_map, sanitize_function, typemap, value_to_seq
from .io import _py_dict_to_cntk_dict

__doc__ = '''\
A training session encapsulates a typical training loop and binds together a minibatch source that is used for training, a :doc:`trainer <cntk.trainer>` and an optional cross validation minibatch source. A training session takes care of consistent checkpointing and progress printing with specified frequencies. 
'''


class TrainingSession(cntk_py.TrainingSession):
    '''
    The instance of the class should be created by using :func:`~cntk.training_session.training_session` function.

    A training session trains a model using the specified ``trainer`` and the ``training_minibatch_source``
    where the minibatch size defined by ``mb_size_schedule``. The mapping between the input variables and the
    corresponding input streams should be specified using ``model_inputs_to_mb_source_mapping``.
    The size of the training set can be controlled either during creation of the training minibatch 
    source or using ``max_training_samples`` parameter. 
    Checkpointing is done both for the trainer and the training minibatch source.
    Progress printing happens each ``progress_frequency`` samples using the provided ``progress_printer``. 

    Args:
        training_minibatch_source (:class:`~cntk.io.MinibatchSource`): minibatch source used for training
        trainer (:class:`~cntk.trainer.Trainer`): trainer
        mb_size_schedule (:class:`~cntk.cntk_py.minibatch_size_schedule`): minibatch schedule for training
        progress_printer (:class:`~cntk.utils.progress_print.ProgressPrinter`): progress printer
        model_inputs_to_mb_source_mapping (dict): mapping between input variables and input streams
        checkpoint_frequency (int): checkpoint frequency in samples. If 0, no checkpointing takes place. 
          If ``sys.maxsize``, a single checkpoint is taken at the end of the training.
        checkpoint_filename (str): checkpoint file name.
        save_all_checkpoints (bool): saves all checkpoints, using ``checkpoint_filename`` as prefix and checkpoint index as a suffix.
        restore (bool): flag, indicating whether to restore from available checkpoint before the start of the training
        progress_frequency (int): frequency in samples for aggregated progress printing
        cv_source (:class:`~cntk.io.MinibatchSource`): minibatch source used for cross validation
        cv_frequency (int): frequency in samples for cross validation
          If ``sys.maxsize``, a single cross validation is performed at the end of training.
        cv_mb_size_schedule (:class:`~cntk.cntk_py.minibatch_size_schedule`): minibatch schedule for cross validation
        max_training_samples (int): maximum number of samples used for training
    '''

    def __init__(self, training_minibatch_source, trainer, mb_size_schedule,
                 progress_printer, model_inputs_to_mb_source_mapping,
                 checkpoint_frequency, checkpoint_filename, save_all_checkpoints,
                 restore, progress_frequency, cv_source, cv_frequency, cv_mb_size_schedule, max_training_samples):

        self.progress_printer = progress_printer
        self.trainer = trainer

        if not isinstance(mb_size_schedule, cntk_py.minibatch_size_schedule):
            raise ValueError('mb_size_schedule type (%s) not supported. '
                             'mb_size_schedule must be a schedule '
                             '(output of minibatch_size_schedule() function)'
                             % type(mb_size_schedule))

        if checkpoint_filename is None:
            if checkpoint_frequency is not None and checkpoint_frequency != 0:
                raise ValueError(
                    "Checkpoint frequency cannot be specified without checkpoint_filename")
            checkpoint_frequency = 0
            checkpoint_filename = ""

        if progress_frequency is None:
            progress_frequency = sys.maxsize

        if cv_source is None:
            if cv_frequency is not None and cv_frequency != 0:
                raise ValueError(
                    "Cross validation frequency cannot be specified without cross validation minibatch source")
            cv_frequency = 0

        if cv_frequency is None:
            cv_frequency = sys.maxsize

        if max_training_samples is None:
            max_training_samples = sys.maxsize

        if checkpoint_frequency is None:
            checkpoint_frequency = sys.maxsize

        if cv_mb_size_schedule is None:
            cv_mb_size_schedule = minibatch_size_schedule(1)

        super(TrainingSession, self).__init__(
            training_minibatch_source,
            trainer,
            model_inputs_to_mb_source_mapping,
            mb_size_schedule,
            checkpoint_frequency,
            checkpoint_filename,
            cv_source,
            cv_mb_size_schedule,
            cv_frequency,
            restore,
            save_all_checkpoints,
            max_training_samples,
            progress_frequency)

    @typemap
    def train(self, device=None):
        '''
        Perform training on a specified device.

        Args:
            device (:class:~cntk.device.DeviceDescriptor): the device descriptor containing
               the type and id of the device where training takes place.
        '''

        if not device:
            device = use_default_device()

        super(TrainingSession, self).train(device)

    def on_minibatch_end(self):
        '''
        Callback that gets executed at the end of each minibatch.
        '''
        if self.progress_printer and self.trainer.total_number_of_samples_seen != 0:
            self.progress_printer.update_with_trainer(
                self.trainer, with_metric=True)

    def on_progress(self, index):
        '''
        Callback that gets executed with the ``progress_frequency`` frequency in samples.

        Args:
            index (int): index of the current callback.
        '''
        if self.progress_printer:
            self.progress_printer.epoch_summary(with_metric=True)

    def on_cross_validation_end(self, index, average_error, num_samples, num_minibatches):
        '''
        Callback that gets executed at the end of cross validation.

        Args:
            index (int): index of the current callback.
            average_error (float): average error for the cross validation
            num_samples (int): number of samples in cross validation
            num_minibatches (int): number of minibatch in cross validation
        '''
        if self.progress_printer:
            msg = "Cross Validation [{}]: Minibatch[1-{}]: errs = {:0.2f}% * {}".format(
                index + 1, num_minibatches, average_error * 100, num_samples)
            self.progress_printer.log(msg)


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
def training_session(training_minibatch_source,
                     trainer, mb_size_schedule,
                     progress_printer=None,
                     model_inputs_to_mb_source_mapping={},
                     checkpoint_filename=None,
                     checkpoint_frequency=None,
                     save_all_checkpoints=False,
                     restore=True,
                     progress_frequency=None,
                     cv_source=None,
                     cv_mb_size_schedule=None,
                     cv_frequency=None,
                     max_training_samples=None):
    '''
    A factory function to create a training session object.

    Args: 
        training_minibatch_source (:class:`~cntk.io.MinibatchSource`): minibatch source used for training
        trainer (:class:`~cntk.trainer.Trainer`): trainer
        mb_size_schedule (:class:`~cntk.cntk_py.minibatch_size_schedule`): minibatch schedule for training
        progress_printer (:class:`~cntk.utils.progress_print.ProgressPrinter`): progress printer
        model_inputs_to_mb_source_mapping (dict): mapping between input variables and input streams
        checkpoint_filename (str): checkpoint file name.
        checkpoint_frequency (int): checkpoint frequency in samples. If 0, no checkpointing takes place. 
          If ``sys.maxsize``, a single checkpoint is taken at the end of the training.
        save_all_checkpoints (bool): saves all checkpoints, using ``checkpoint_filename`` as prefix and checkpoint index as a suffix.
        restore (bool): flag, indicating whether to restore from available checkpoint before the start of the training
        progress_frequency (int): frequency in samples for aggregated progress printing
        cv_source (:class:`~cntk.io.MinibatchSource`): minibatch source used for cross validation
        cv_frequency (int): frequency in samples for cross validation
        cv_mb_size_schedule (:class:`~cntk.cntk_py.minibatch_size_schedule`): minibatch schedule for cross validation
          If ``sys.maxsize``, a single cross validation is performed at the end of training.
        max_training_samples (int): maximum number of samples used for training

    Returns:
        Instance of :class:`~TrainingSession`
    '''
    return TrainingSession(training_minibatch_source, trainer,
                           mb_size_schedule, progress_printer,
                           model_inputs_to_mb_source_mapping,
                           checkpoint_frequency,
                           checkpoint_filename,
                           save_all_checkpoints=save_all_checkpoints,
                           restore=restore,
                           progress_frequency=progress_frequency,
                           cv_source=cv_source,
                           cv_frequency=cv_frequency,
                           cv_mb_size_schedule=cv_mb_size_schedule,
                           max_training_samples=max_training_samples)
