# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import sys
from . import cntk_py
from .device import use_default_device
from .utils import sanitize_var_map, sanitize_function, typemap, value_to_seq
from .io import _py_dict_to_cntk_dict

__doc__= '''\
A training session encapsulates a typical training loop and binds together the minibatch source, the :doc:`trainer <cntk.trainer>` and checkpointing.
'''

class TrainingSession(cntk_py.TrainingSession):
    '''
    A training session is an abstraction that encapsulates a typical training loop given
    a minibatch source and a :doc:`trainer <cntk.trainer>` and takes care of checkpointing.
    '''
    def __init__(self, training_minibatch_source, trainer, mb_size_schedule,
                 progress_printer, model_inputs_to_mb_source_mapping, 
                 checkpoint_frequency, checkpoint_filename, save_all_checkpoints, 
                 restore, progress_frequency, cv_source, cv_frequency, cv_mb_size_schedule, max_training_samples):

        self.progress_printer = progress_printer
        self.trainer=trainer       

        super(TrainingSession, self).__init__ (
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
        Performs training.
        '''

        if not device:
            device = use_default_device()

        super(TrainingSession, self).train(device)

    def on_minibatch_end(self):
        if self.progress_printer and self.trainer.total_number_of_samples_seen != 0:
            self.progress_printer.update_with_trainer(self.trainer, with_metric=True)

    def on_progress(self, index):
        if self.progress_printer:
            self.progress_printer.epoch_summary(with_metric=True)

    def on_cross_validation_end(self, index, average_error, num_samples, num_minibatches):
        if self.progress_printer:
            msg = "Cross Validation [{}]: Minibatch[1-{}]: errs = {:0.2f}% * {}".format(index + 1, num_minibatches, average_error * 100, num_samples)
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

    raise ValueError('schedule must be either a float or a list, not %s'%type(schedule))

@typemap
def training_session(training_minibatch_source,
                     trainer, mb_size_schedule,
                     progress_printer = None,
                     model_inputs_to_mb_source_mapping = {},
                     checkpoint_filename = None,
                     checkpoint_frequency = None,
                     save_all_checkpoints = False,
                     restore = True,
                     progress_frequency = None,
                     cv_source = None,
                     cv_mb_size_schedule = None,
                     cv_frequency = None,
                     max_training_samples = None):
    '''
    Creates a basic training session.

    Args:
        training_minibatch_source: a minibatch source that will be used for training.
        trainer: a Trainer.
        mb_size_schedule: a minibatch size schedule for training. Created using :func:`minibatch_size_schedule`
        progress_printer: a progress printer instance
        model_inputs_to_mb_source_mapping: mapping between the input node names of the model and the stream 
         names provided from the minibatch source. By default all streams are taken with their respective names.
        checkpoint_filename: a file name of the checkpoint file, if None, the checkpointing is disabled.
        checkpoint_frequency: an approximate number of global samples processed accross the workers 
         after which the checkpoint is taken. Should be positive number if the checkpoint file is specified.
        save_all_checkpoints: flag, indicating whether to store all checkpoints, by default only the last checkpoint is preserved
        restore: flag, indicating whether perform restore of the training session from the checkpoint before the start of the training
        progress_frequency: an approximate number of global samples processed accross the workers 
         after which the summary of metrics is reported using the progress_printer
        cv_source: a minibatch source that will be used for cross validation.
        cv_mb_size_schedule: a minibatch size schedule for cross validation. Created using :func:`minibatch_size_schedule`
        progress_frequency: an approximate number of global samples processed accross the workers 
         after which the cross validation takes place
        max_training_samples: max number of samples after which the training should be stopped

    Returns:
        Instance of a :class:`TrainingSession`
    '''
    if not isinstance(mb_size_schedule, cntk_py.minibatch_size_schedule):
        raise ValueError('mb_size_schedule type (%s) not supported. '
                         'mb_size_schedule must be a schedule '
                         '(output of minibatch_size_schedule() function)' 
                         % type(mb_size_schedule))

    if checkpoint_filename is None:
        if checkpoint_frequency is not None and checkpoint_frequency != 0:
            raise ValueError("Checkpoint frequency cannot be specified without checkpoint_filename")
        checkpoint_frequency = 0
        checkpoint_filename=""

    if progress_frequency is None:
        progress_frequency = sys.maxsize

    if cv_source is None:
        if cv_frequency is not None and cv_frequency != 0:
            raise ValueError("Cross validation frequency cannot be specified without cross validation minibatch source")
        cv_frequency = 0

    if cv_frequency is None:
        cv_frequency = sys.maxsize

    if max_training_samples is None:
        max_training_samples = sys.maxsize

    if checkpoint_frequency is None:
        checkpoint_frequency = sys.maxsize

    if cv_mb_size_schedule is None:
        cv_mb_size_schedule = minibatch_size_schedule(1)

    return TrainingSession(training_minibatch_source, trainer, 
                           mb_size_schedule, progress_printer, 
                           model_inputs_to_mb_source_mapping, 
                           checkpoint_frequency,
                           checkpoint_filename,
                           save_all_checkpoints=save_all_checkpoints,
                           restore=restore,
                           progress_frequency=progress_frequency,
                           cv_source=cv_source,
                           cv_mb_size_schedule=cv_mb_size_schedule,
                           cv_frequency=cv_frequency,
                           max_training_samples=max_training_samples)
