# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================
from __future__ import print_function
import os
import sys
import time

from cntk import cntk_py

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
    Allows printing various training time statistics (e.g. loss and metric) and printing them as training progresses.
    '''

    def __init__(self, freq=None, first=0, tag='', log_to_file=None, rank=None, gen_heartbeat=False, num_epochs=300,
                 test_freq=None, test_first=0):
        '''
        Constructor.

        Args:
            freq (`int` or `None`, default `None`):  determines how often
              printing will occur. The value of 0 means an geometric
              schedule (1,2,4,...). A value > 0 means a arithmetic schedule
              (a log print for minibatch number: ``freq``, a log print for minibatch number: 2*``freq``,
              a log print for minibatch number: 3*``freq``,...), and a value of None means no per-minibatch log.
            first (`int`, default 0): Only start logging after the minibatch number is greater or equal to ``first``.
            tag (`string`, default EmptyString): prepend minibatch log lines with your own string
            log_to_file (`string` or `None`, default `None`): if None, output log data to stdout.
              If a string is passed, the string is path to a file for log data.
            rank (`int` or `None`, default `None`): set this to distributed.rank if you are using distributed
              parallelism -- each rank's log will go to separate file.
            gen_heartbeat (`bool`, default `False`): If True output a progress message every 10 seconds or so to stdout.
            num_epochs (`int`, default 300): The total number of epochs to be trained.  Used for some metadata.
              This parameter is optional.
            test_freq (`int` or `None`, default `None`): similar to ``freq``, but applies to printing intermediate
              test results.
            test_first (`int`, default 0): similar to ``first``, but applies to printing intermediate test results.
        '''
        if freq is None:
            freq = sys.maxsize

        if test_freq is None:
            test_freq = sys.maxsize

        super(ProgressPrinter, self).__init__(freq, first, test_freq, test_first)

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

        # print out data about CNTK build
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

            self.___logprint('CNTKCommandTrainInfo: train : ' + str(num_epochs))
            self.___logprint('CNTKCommandTrainInfo: CNTKNoMoreCommands_Total : ' + str(num_epochs))
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
        DEPRECATED. Use :func:`cntk.utils.ProgressPrinter.update_training` instead.

        Update the current loss, the minibatch size and optionally the metric using the information from the
        ``trainer``.

        Args:
            trainer (:class:`cntk.trainer.Trainer`): trainer from which information is gathered
            with_metric (`bool`): whether to update the metric accumulators
        '''
        if self.total_updates == 0:
            # Only warn once to avoid flooding with warnings.
            _warn_deprecated('Use ProgressPrinter.update_progress() instead.')

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
                format_str += 'metric = {:0.2f}% * {:d}'
                format_args.extend([_avg(aggregate_metric, samples) * 100.0, samples[1] - samples[0]])

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

        if aggregate_metric is not None:
            avg_metric = _avg(aggregate_metric, samples)
            msg = "Finished Epoch[{} of {}]: {}loss = {:0.6f} * {}, metric = {:0.2f}% * {} {:0.3f}s ({:5.1f} samples per second);".format(
                summaries, self.num_epochs, self.tag, avg_loss, samples, avg_metric * 100.0, samples,
                elapsed_seconds, speed)
        else:
            msg = "Finished Epoch[{} of {}]: {}loss = {:0.6f} * {} {:0.3f}s ({:5.1f} samples per second);".format(
                summaries, self.num_epochs, self.tag, avg_loss, samples, elapsed_seconds, speed)

        self.___logprint(msg)

    def on_write_test_summary(self, samples, updates, summaries, aggregate_metric, elapsed_milliseconds):
        # Override for ProgressWriter.on_write_test_summary.
        self.___logprint("Finished Evaluation [{}]: Minibatch[1-{}]: metric = {:0.2f}% * {};".format(
            summaries, updates, _avg(aggregate_metric, samples) * 100.0, samples))


class TensorBoardProgressWriter(cntk_py.ProgressWriter):
    '''
    Allows tracking various training time statistics (e.g. loss and metric) and write them as TensorBoard event files.
    The generated files can be opened in TensorBoard to visualize the progress.
    '''

    def __init__(self, freq=None, log_dir='.', rank=None, model=None):
        '''
        Constructor.

        Args:
            freq (`int` or `None`, default `None`): frequency at which progress is logged.
              For example, the value of 2 will cause the progress to be logged every second time when
              `:func:cntk.util.TensorBoardFileWriter.update_with_trainer` is invoked.
              None indicates that progress is logged only when
              `:func:cntk.util.TensorBoardFileWriter.summarize_progress` is invoked.
              Must be a positive integer otherwise.
            log_dir (`string`, default '.'): directory where to create a TensorBoard event file.
            rank (`int` or `None`, default `None`): rank of a worker when using distributed training, or `None` if
             training locally. If not `None`, event files will be created in log_dir/rank[rank] rather than log_dir.
            model (:class:`cntk.ops.Function` or `None`, default `None`): model graph to plot.
        '''
        if freq is None:
            freq = sys.maxsize

        super(TensorBoardProgressWriter, self).__init__(freq, 0, sys.maxsize, 0)

        # Only log either when rank is not specified or when rank is 0.
        self.writer = cntk_py.TensorBoardFileWriter(log_dir, model) if not rank else None
        self.closed = False

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
            self.write_value('summary/test_avg_metric', avg_metric, self.summaries)


# print the total number of parameters to log
def log_number_of_parameters(model, trace_level=0):
    parameters = model.parameters
    from functools import reduce
    from operator import add, mul
    total_parameters = reduce(add, [reduce(mul, p1.shape) for p1 in parameters], 0)
    # BUGBUG: If model has uninferred dimensions, we should catch that and fail here
    print("Training {} parameters in {} parameter tensors.".format(total_parameters, len(parameters)))
    if trace_level > 0:
        print()
        for p in parameters:
            print("\t{}".format(p.shape))