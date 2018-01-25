# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================
from __future__ import print_function
from __future__ import division

import sys
import time

from cntk import cntk_py, core
from ..device import cpu 

def _warn_deprecated(message):
    from warnings import warn
    warn('DEPRECATED: ' + message, DeprecationWarning, stacklevel=2)


def _avg(numerator, denominator):
    if isinstance(numerator, tuple):
        numerator = numerator[1] - numerator[0]
    if isinstance(denominator, tuple):
        denominator = denominator[1] - denominator[0]
    return (numerator / denominator) if denominator > 0 else 0.0


# TODO: Let's switch to import logging in the future instead of print. [ebarsoum]
class ProgressPrinter(cntk_py.ProgressWriter):
    '''
    Allows printing various statistics (e.g. loss and metric) as training/evaluation progresses.

    Args:
        freq (`int` or `None`, default `None`):  determines how often printing of training progress will occur.
          A value of 0 means a geometric schedule (1,2,4,...).
          A value > 0 means an arithmetic schedule (print for minibatch number: ``freq``,
          print for minibatch number: ``2 * freq``, print for minibatch number: ``3 * freq``,...).
          A value of None means no per-minibatch log.
        first (`int`, default 0): Only start printing after the training minibatch number is greater or equal to
          ``first``.
        tag (`string`, default EmptyString): prepend minibatch log lines with your own string
        log_to_file (`string` or `None`, default `None`): if None, output log data to stdout.
          If a string is passed, the string is path to a file for log data.
        rank (`int` or `None`, default `None`): set this to distributed.rank if you are using distributed
          parallelism -- each rank's log will go to separate file.
        gen_heartbeat (`bool`, default `False`): If True output a progress message every 10 seconds or so to stdout.
        num_epochs (`int`, default None): The total number of epochs to be trained.  Used for some metadata.
          This parameter is optional.
        test_freq (`int` or `None`, default `None`): similar to ``freq``, but applies to printing intermediate
          test results.
        test_first (`int`, default 0): similar to ``first``, but applies to printing intermediate test results.
        metric_is_pct (`bool`, default True): Treat metric as a percentage for output purposes.
        distributed_freq (`int` or `None`, default `None`): similar to ``freq``, but applies to printing distributed-training 
          worker synchronization info.
        distributed_first (`int`, default 0): similar to ``first``, but applies to printing distributed-training 
          worker synchronization info.
    '''

    def __init__(self, freq=None, first=0, tag='', log_to_file=None, rank=None, gen_heartbeat=False, num_epochs=None,
                 test_freq=None, test_first=0, metric_is_pct=True, distributed_freq=None, distributed_first=0):
        '''
        Constructor.
        '''
        if freq is None:
            freq = sys.maxsize

        if test_freq is None:
            test_freq = sys.maxsize

        if distributed_freq is None:
            distributed_freq = sys.maxsize

        super(ProgressPrinter, self).__init__(freq, first, test_freq, test_first, distributed_freq, distributed_first)

        self.loss_since_start = 0
        self.metric_since_start = 0
        self.samples_since_start = 0
        self.updates_since_start = 0
        self.loss_since_last = 0
        self.metric_since_last = 0
        self.samples_since_last = 0
        self.total_updates = 0
        self.epochs = 0
        self.freq = freq
        self.first = first
        self.test_freq = test_freq
        self.tag = '' if not tag else "[{}] ".format(tag)
        self.epoch_start_time = time.time()
        self.progress_timer_time = 0
        self.log_to_file = log_to_file
        self.gen_heartbeat = gen_heartbeat
        self.num_epochs = num_epochs
        self.metric_is_pct = metric_is_pct
        if metric_is_pct:
            self.metric_multiplier = 100.0
        else:
            self.metric_multiplier = 1.0

        self.__disown__()

        # print out data about CNTK build
        # TODO: this is for internal purposes, so find better way
        cntk_py.print_built_info()

        self.logfilename = None
        if self.log_to_file is not None:
            self.logfilename = self.log_to_file

            if rank is not None:
                self.logfilename = self.logfilename + 'rank' + str(rank)

            # print to stdout
            print("Redirecting log to file " + self.logfilename)

            with open(self.logfilename, "w") as logfile:
                logfile.write(self.logfilename + "\n")

            self.___logprint('CNTKCommandTrainInfo: train : ' + str(num_epochs if num_epochs is not None else 300))
            self.___logprint('CNTKCommandTrainInfo: CNTKNoMoreCommands_Total : ' + str(num_epochs if num_epochs is not None else 300))
            self.___logprint('CNTKCommandTrainBegin: train')

        if freq == 0:
            self.___logprint(' average      since    average      since      examples')
            self.___logprint('    loss       last     metric       last              ')
            self.___logprint(' ------------------------------------------------------')

    def end_progress_print(self, msg=""):
        '''
        Prints the given message signifying the end of training.

        Args:
            msg (`string`, default ''): message to print.
        '''
        self.___logprint('CNTKCommandTrainEnd: train')
        if msg != "" and self.log_to_file is not None:
            self.___logprint(msg)

    def log(self, message):
        '''
        Prints any message the user wishes to place in the log.

        Args:
            msg (`string`): message to print.
        '''
        self.___logprint(message)

    def avg_loss_since_start(self):
        '''
        DEPRECATED.

        Returns: the average loss since the start of accumulation
        '''
        _warn_deprecated('The method was deprecated.')
        return _avg(self.loss_since_start, self.samples_since_start)

    def avg_metric_since_start(self):
        '''
        DEPRECATED.

        Returns: the average metric since the start of accumulation
        '''
        _warn_deprecated('The method was deprecated.')
        return _avg(self.metric_since_start, self.samples_since_start)

    def avg_loss_since_last(self):
        '''
        DEPRECATED.

        Returns: the average loss since the last print
        '''
        _warn_deprecated('The method was deprecated.')
        return _avg(self.loss_since_last, self.samples_since_last)

    def avg_metric_since_last(self):
        '''
        DEPRECATED.

        Returns: the average metric since the last print
        '''
        _warn_deprecated('The method was deprecated.')
        return _avg(self.metric_since_last, self.samples_since_last)

    def reset_start(self):
        '''
        DEPRECATED.

        Resets the 'start' accumulators

        Returns: tuple of (average loss since start, average metric since start, samples since start)
        '''
        _warn_deprecated('The method was deprecated.')
        ret = self.avg_loss_since_start(), self.avg_metric_since_start(), self.samples_since_start
        self.loss_since_start = 0
        self.metric_since_start = 0
        self.samples_since_start = 0
        self.updates_since_start = 0
        return ret

    def reset_last(self):
        '''
        DEPRECATED.

        Resets the 'last' accumulators

        Returns: tuple of (average loss since last, average metric since last, samples since last)
        '''
        if self.total_updates == 0:
            # Only warn once to avoid flooding with warnings.
            _warn_deprecated('The method was deprecated.')
        ret = self.avg_loss_since_last(), self.avg_metric_since_last(), self.samples_since_last
        self.loss_since_last = 0
        self.metric_since_last = 0
        self.samples_since_last = 0
        return ret

    def write(self, key, value):
        # Override for ProgressWriter.write method.
        self.___logprint("{}: {}".format(key, value))

    def ___logprint(self, logline):
        if self.log_to_file == None:
            # to stdout.  if distributed, all ranks merge output into stdout
            print(logline)
        else:
            # to named file.  if distributed, one file per rank
            with open(self.logfilename, "a") as logfile:
                logfile.write(logline + "\n")

    def epoch_summary(self, with_metric=False):
        '''
        DEPRECATED.

        If on an arithmetic schedule print an epoch summary using the 'start' accumulators.
        If on a geometric schedule does nothing.

        Args:
            with_metric (`bool`): if `False` it only prints the loss, otherwise it prints both the loss and the metric
        '''
        _warn_deprecated('The method was deprecated.')
        self.epochs += 1
        epoch_end_time = time.time()
        elapsed_milliseconds = (epoch_end_time - self.epoch_start_time) * 1000
        self.epoch_start_time = epoch_end_time # resetting starttime for use in the next epoch

        metric_since_start = self.metric_since_start if with_metric else None
        self.on_write_training_summary(self.samples_since_start, self.updates_since_start, self.epochs,
                                       self.loss_since_start, metric_since_start, elapsed_milliseconds)

        if self.freq > 0:
            return self.reset_start()

    def ___generate_progress_heartbeat(self):
        timer_delta = time.time() - self.progress_timer_time

        # print progress no sooner than 10s apart
        if timer_delta > 10 and self.gen_heartbeat:
            # print to stdout
            print("PROGRESS: 0.00%")
            self.progress_timer_time = time.time()

    def update(self, loss, minibatch_size, metric=None):
        '''
        DEPRECATED.

        Updates the accumulators using the loss, the minibatch_size and the optional metric.

        Args:
            loss (`float`): the value with which to update the loss accumulators
            minibatch_size (`int`): the value with which to update the samples accumulator
            metric (`float` or `None`): if `None` do not update the metric
             accumulators, otherwise update with the given value
        '''
        if self.total_updates == 0:
            # Only warn once to avoid flooding with warnings.
            _warn_deprecated('The method was deprecated.')

        if minibatch_size == 0:
            return

        self.samples_since_start += minibatch_size
        self.samples_since_last += minibatch_size
        self.loss_since_start += loss * minibatch_size
        self.loss_since_last += loss * minibatch_size
        self.updates_since_start += 1
        self.total_updates += 1

        if metric is not None:
            self.metric_since_start += metric * minibatch_size
            self.metric_since_last += metric * minibatch_size

        self.___generate_progress_heartbeat()

        if ((self.freq == 0 and (self.updates_since_start + 1) & self.updates_since_start == 0) or
            self.freq > 0 and (self.updates_since_start % self.freq == 0 or self.updates_since_start <= self.first)):

            samples = (self.samples_since_start - self.samples_since_last, self.samples_since_start)
            updates = None
            if self.freq > 0:
                if self.updates_since_start <= self.first:  # printing individual MBs
                    first_update = self.updates_since_start
                else:
                    first_update = max(self.updates_since_start - self.freq, self.first)
                updates = (first_update, self.updates_since_start)

            aggregate_loss = (self.loss_since_start - self.loss_since_last, self.loss_since_start)
            aggregate_metric = None
            if metric is not None:
                aggregate_metric = (self.metric_since_start - self.metric_since_last, self.metric_since_start)

            self.on_write_training_update(samples, updates, aggregate_loss, aggregate_metric)
            self.reset_last()

    def update_with_trainer(self, trainer, with_metric=False):
        '''
        DEPRECATED.

        Update the current loss, the minibatch size and optionally the metric using the information from the
        ``trainer``.

        Args:
            trainer (:class:`cntk.train.trainer.Trainer`): trainer from which information is gathered
            with_metric (`bool`): whether to update the metric accumulators
        '''
        if self.total_updates == 0:
            # Only warn once to avoid flooding with warnings.
            _warn_deprecated('Inefficient. '
                             'Please pass an instance of ProgressPrinter to Trainer upon construction.')

        if trainer is not None and trainer.previous_minibatch_sample_count != 0:
            self.update(
                trainer.previous_minibatch_loss_average,
                trainer.previous_minibatch_sample_count,
                trainer.previous_minibatch_evaluation_average if with_metric else None)

    def on_write_training_update(self, samples, updates, aggregate_loss, aggregate_metric):
        # Override for ProgressWriter.on_write_training_update.
        self.___write_progress_update(samples, updates, aggregate_loss, aggregate_metric, self.freq, '')

    def on_training_update_end(self):
        # Override for ProgressWriter.on_training_update_end.
        self.___generate_progress_heartbeat()

    def on_write_test_update(self, samples, updates, aggregate_metric):
        # Override for ProgressWriter.on_write_test_update.
        self.___write_progress_update(samples, updates, None, aggregate_metric, self.test_freq, 'Evaluation ')

    def on_write_distributed_sync_update(self, samples, updates, aggregate_metric):
        # Override for ProgressWriter.on_write_distributed_sync_update.
        self.___logprint("Distributed training: #Syncs elapsed = {}, #Samples elapsed = {}".format(updates[1] - updates[0], samples[1] - samples[0]))

    def ___write_progress_update(self, samples, updates, aggregate_loss, aggregate_metric, frequency, name):
        format_str = ' '
        format_args = []

        if frequency == 0:
            if aggregate_loss is not None:
                format_str += '{:8.3g}   {:8.3g}   '
                format_args.extend([_avg(aggregate_loss[1], samples[1]), _avg(aggregate_loss, samples)])
            else:
                format_str += '{:8s}   {:8s}   '
                format_args.extend(['', ''])

            if aggregate_metric is not None:
                format_str += '{:8.3g}   {:8.3g}   '
                format_args.extend([_avg(aggregate_metric[1], samples[1]), _avg(aggregate_metric, samples)])
            else:
                format_str += '{:8s}   {:8s}   '
                format_args.extend(['', ''])

            format_str += ' {:10d}'
            format_args.append(samples[1])
        else:
            format_str += '{}Minibatch[{:4d}-{:4d}]: '
            format_args.extend([name, updates[0] + 1, updates[1]])

            if aggregate_loss is not None:
                format_str += 'loss = {:0.6f} * {:d}'
                format_args.extend([_avg(aggregate_loss, samples), samples[1] - samples[0]])

            if aggregate_metric is not None:
                if aggregate_loss is not None:
                    format_str += ', '
                if self.metric_is_pct:
                    format_str += 'metric = {:0.2f}% * {:d}'
                else:
                    format_str += 'metric = {:0.6f} * {:d}'

                format_args.extend([_avg(aggregate_metric, samples) * self.metric_multiplier, samples[1] - samples[0]])

            format_str += ';'

        self.___logprint(format_str.format(*format_args))

    def on_write_training_summary(self, samples, updates, summaries, aggregate_loss, aggregate_metric,
                                  elapsed_milliseconds):
        # Override for ProgressWriter.on_write_training_summary.
        if self.freq == 0:
            # Only log training summary when on arithmetic schedule.
            return

        elapsed_seconds = elapsed_milliseconds / 1000
        speed = _avg(samples, elapsed_seconds)
        avg_loss = _avg(aggregate_loss, samples)

        of_epochs = " of " + str(self.num_epochs) if self.num_epochs is not None else ''
        if aggregate_metric is not None:
            avg_metric = _avg(aggregate_metric, samples)
            if self.metric_is_pct:
                fmt_str = "Finished Epoch[{}{}]: {}loss = {:0.6f} * {}, metric = {:0.2f}% * {} {:0.3f}s ({:5.1f} samples/s);"
            else:
                fmt_str = "Finished Epoch[{}{}]: {}loss = {:0.6f} * {}, metric = {:0.6f} * {} {:0.3f}s ({:5.1f} samples/s);"
            msg = fmt_str.format(summaries, of_epochs, self.tag, avg_loss, samples, avg_metric * self.metric_multiplier,
                    samples, elapsed_seconds, speed)
        else:
            msg = "Finished Epoch[{}{}]: {}loss = {:0.6f} * {} {:0.3f}s ({:5.1f} samples/s);".format(
                summaries, of_epochs, self.tag, avg_loss, samples, elapsed_seconds, speed)

        self.___logprint(msg)

    def on_write_test_summary(self, samples, updates, summaries, aggregate_metric, elapsed_milliseconds):
        # Override for ProgressWriter.on_write_test_summary.
        if self.metric_is_pct:
            fmt_str = "Finished Evaluation [{}]: Minibatch[1-{}]: metric = {:0.2f}% * {};"
        else:
            fmt_str = "Finished Evaluation [{}]: Minibatch[1-{}]: metric = {:0.6f} * {};"
        self.___logprint(fmt_str.format(summaries, updates,
                            _avg(aggregate_metric, samples) * self.metric_multiplier, samples))


class TensorBoardProgressWriter(cntk_py.ProgressWriter):
    '''
    Allows writing various statistics (e.g. loss and metric) to TensorBoard event files during training/evaluation.
    The generated files can be opened in TensorBoard to visualize the progress.

    Args:
        freq (`int` or `None`, default `None`): frequency at which training progress is written.
          None indicates that progress is logged only at the end of training.
          Must be a positive integer otherwise.
        log_dir (`string`, default '.'): directory where to create a TensorBoard event file.
        rank (`int` or `None`, default `None`): rank of a worker when using distributed training, or `None` if
         training locally. If not `None`, event files will be created only by rank 0.
        model (:class:`cntk.ops.functions.Function` or `None`, default `None`): model graph to plot.
    '''

    def __init__(self, freq=None, log_dir='.', rank=None, model=None):
        '''
        Constructor.
        '''
        if freq is None:
            freq = sys.maxsize

        super(TensorBoardProgressWriter, self).__init__(freq, 0, sys.maxsize, 0, sys.maxsize, 0)

        # Only log either when rank is not specified or when rank is 0.
        self.writer = cntk_py.TensorBoardFileWriter(log_dir, model) if not rank else None
        self.closed = False
        self.__disown__()

    def write_value(self, name, value, step):
        '''
        Record value of a scalar variable at the given time step.

        Args:
            name (`string`): name of a variable.
            value (`float`): value of the variable.
            step (`int`): time step at which the value is recorded.
        '''
        if self.closed:
            raise RuntimeError('Attempting to use a closed TensorBoardProgressWriter')

        if self.writer:
            self.writer.write_value(str(name), float(value), int(step))

    def write_image(self, name, data, step):
        if self.closed:
            raise RuntimeError('Attempting to use a closed TensorBoardProgressWriter')

        if self.writer:
            for k in data:
                value = core.Value._as_best_data_type(k, data[k])
                ndav = core.NDArrayView.from_data(value, cpu())
                self.writer.write_image(str(name), ndav, int(step))

    def flush(self):
        '''Make sure that any outstanding records are immediately persisted.'''
        if self.closed:
            raise RuntimeError('Attempting to use a closed TensorBoardProgressWriter')

        if self.writer:
            self.writer.flush()

    def close(self):
        '''
        Make sure that any outstanding records are immediately persisted, then close any open files.
        Any subsequent attempt to use the object will cause a RuntimeError.
        '''
        if self.closed:
            raise RuntimeError('Attempting to use a closed TensorBoardProgressWriter')

        if self.writer:
            self.writer.close()
            self.closed = True

    def on_write_training_update(self, samples, updates, aggregate_loss, aggregate_metric):
        # Override for ProgressWriter.on_write_training_update().
        self.write_value('minibatch/avg_loss', _avg(aggregate_loss, samples), self.total_training_updates())
        self.write_value('minibatch/avg_metric', _avg(aggregate_metric, samples), self.total_training_updates())

    def on_write_test_update(self, samples, updates, aggregate_metric):
        # Override for ProgressWriter.on_write_test_update().
        # It is not particularly useful to record per-minibatch test results in TensorBoard,
        # hence it is not currently supported.
        raise NotImplementedError(
            'TensorBoardProgressWriter does not support recording per-minibatch cross-validation results')

    def on_write_training_summary(self, samples, updates, summaries, aggregate_loss, aggregate_metric,
                                  elapsed_milliseconds):
        # Override for BaseProgressWriter.on_write_training_summary().
        self.write_value('summary/avg_loss', _avg(aggregate_loss, samples), summaries)
        self.write_value('summary/avg_metric', _avg(aggregate_metric, samples), summaries)

    def on_write_test_summary(self, samples, updates, summaries, aggregate_metric, elapsed_milliseconds):
        # Override for BaseProgressWriter.on_write_test_summary().
        avg_metric = _avg(aggregate_metric, samples)
        if self.total_training_updates() != 0:
            # Record test summary using training minibatches as a step.
            # This allows to easier correlate the training and test metric graphs in TensorBoard.
            self.write_value('minibatch/test_avg_metric', avg_metric, self.total_training_updates())
        else:
            self.write_value('summary/test_avg_metric', avg_metric, summaries)


class TrainingSummaryProgressCallback(cntk_py.ProgressWriter):
    '''
    Helper to pass a callback function to be called after each training epoch
    to :class:`~cntk.train.trainer.Trainer`,
    :class:`~cntk.eval.evaluator.Evaluator`, and :class:`~cntk.train.training_session.TrainingSession`,
    as well a :func:`cntk.ops.functions.Function.train`, :func:`cntk.ops.functions.Function.test`.

    This allows the user to add additional logging after each training epoch.

    Args:
     epoch_size (int): periodically call the callback after processing this many samples
     callback (function): function(epoch_index, epoch_loss, epoch_metric, epoch_samples)
    '''
    def __init__(self, epoch_size, callback):
        self._epoch_size = epoch_size
        self._callback = callback
        super(TrainingSummaryProgressCallback, self).__init__(sys.maxsize, 0, epoch_size, 0, sys.maxsize, 0)
        self.__disown__()
    def on_write_training_update(self, samples, updates, aggregate_loss, aggregate_metric):
        pass
    def on_write_test_update(self, *args, **kwargs):
        pass
    def on_write_training_summary(self, samples, updates, summaries, aggregate_loss, aggregate_metric, elapsed_milliseconds):
        self._callback(summaries-1, aggregate_loss, aggregate_metric, samples)
        pass
    def on_write_test_summary(self, samples, updates, summaries, aggregate_metric, elapsed_milliseconds):
        pass
    def write(self, *args, **kwargs):
        pass



# print the total number of parameters to log
def log_number_of_parameters(model, trace_level=0):
    parameters = model.parameters
    from functools import reduce
    from operator import add, mul
    from _cntk_py import InferredDimension
    if any(any(dim == InferredDimension for dim in p.shape) for p in parameters):
        total_parameters = 'so far unspecified number of'
    else:
        total_parameters = sum([reduce(mul, p.shape + (1,)) for p in parameters])
        # the +(1,) is needed so that this works for empty shapes (scalars)
    print("Training {} parameters in {} parameter tensors.".format(total_parameters, len(parameters)))
    if trace_level > 0:
        print()
        for p in parameters:
            print("\t{}".format(p.shape))
