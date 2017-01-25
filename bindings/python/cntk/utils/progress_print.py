# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================
from __future__ import print_function
import os
import time

from cntk.cntk_py import TensorBoardFileWriter


# TODO: Let's switch to import logging in the future instead of print. [ebarsoum]
class ProgressPrinter(object):
    '''
    Allows tracking various training time statistics (e.g. loss and metric)
    and output them as training progresses.

    It provides the number of samples, average loss and average metric
    since the last output or since the start of accumulation.

    Args:
        freq (int or None, default None):  determines how often
         printing will occur. The value of 0 means an geometric
         schedule (1,2,4,...). A value > 0 means a arithmetic schedule
         (a log print for minibatch number: ``freq``, a log print for minibatch number: 2*``freq``, a log print for minibatch number: 3*``freq``,...), and a value of None means no per-minibatch log.
        first (int, default 0): Only start logging after the minibatch number is greater or equal to ``first``.
        tag (string, default EmptyString): prepend minibatch log lines with your own string
        log_to_file (string or None, default None): if None, output log data to stdout.  If a string is passed, the string is path to a file for log data.
        gen_heartbeat (bool, default False): If True output a progress message every 10 seconds or so to stdout.
        num_epochs (int, default 300): The total number of epochs to be trained.  Used for some metadata.  This parameter is optional.
        tensorboard_log_dir (string or None, default None): if a string is passed, logs statistics to the TensorBoard events file in the given directory.
        model (:class:`~cntk.ops.Function` or None, default None): if a Function is passed and ``tensorboard_log_dir`` is not None, records model graph to a TensorBoard events file.
    '''

    def __init__(self, freq=None, first=0, tag='', log_to_file=None, rank=None, gen_heartbeat=False, num_epochs=300,
                 tensorboard_log_dir=None, model=None):
        '''
        Constructor. The optional ``freq`` parameter determines how often
        printing will occur. The value of 0 means an geometric
        schedule (1,2,4,...). A value > 0 means a arithmetic schedule
        (freq, 2*freq, 3*freq,...), and a value of None means no per-minibatch log.
        set log_to_file if you want the output to go file instead of stdout.
        set rank to distributed.rank if you are using distibuted parallelism -- each rank's log will go to seperate file.
        '''
        from sys import maxsize
        if freq is None:
            freq = maxsize

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
        self.tag = '' if not tag else "[{}] ".format(tag)
        self.epoch_start_time = 0
        self.progress_timer_time = 0
        self.log_to_file = log_to_file
        self.rank = rank
        self.gen_heartbeat = gen_heartbeat
        self.num_epochs =  num_epochs

        # Create TensorBoardFileWriter if the path to a log directory was provided.
        self.tensorboard_writer = None
        if tensorboard_log_dir is not None:
            tb_run_name = tag.lower() if tag else ''
            if self.rank is not None:
                tb_run_name += 'rank' + str(self.rank)

            if tb_run_name:
                tensorboard_log_dir = os.path.join(tensorboard_log_dir, tb_run_name)
            self.tensorboard_writer = TensorBoardFileWriter(tensorboard_log_dir, model)

        self.logfilename = None
        if self.log_to_file is not None:
            self.logfilename = self.log_to_file

            if self.rank != None:
                self.logfilename = self.logfilename + 'rank' + str(self.rank)

            # print to stdout
            print("Redirecting log to file " + self.logfilename)

            with open(self.logfilename, "w") as logfile:
                logfile.write(self.logfilename + "\n")

            self.___logprint('CNTKCommandTrainInfo: train : ' + str(num_epochs))
            self.___logprint('CNTKCommandTrainInfo: CNTKNoMoreCommands_Total : ' + str(num_epochs))
            self.___logprint('CNTKCommandTrainBegin: train')

        if freq==0:
            self.___logprint(' average      since    average      since      examples')
            self.___logprint('    loss       last     metric       last              ')
            self.___logprint(' ------------------------------------------------------')

    def end_progress_print(self, msg=""):
        self.___logprint('CNTKCommandTrainEnd: train')
        if msg != "" and self.log_to_file is not None:
            self.___logprint(msg)
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.close()

    def flush(self):
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.flush()

    def avg_loss_since_start(self):
        '''
        Returns: the average loss since the start of accumulation
        '''
        return self.loss_since_start/self.samples_since_start

    def avg_metric_since_start(self):
        '''
        Returns: the average metric since the start of accumulation
        '''
        return self.metric_since_start/self.samples_since_start

    def avg_loss_since_last(self):
        '''
        Returns: the average loss since the last print
        '''
        return self.loss_since_last/self.samples_since_last

    def avg_metric_since_last(self):
        '''
        Returns: the average metric since the last print
        '''
        return self.metric_since_last/self.samples_since_last

    def reset_start(self):
        '''
        Resets the 'start' accumulators

        Returns: tuple of (average loss since start, average metric since start, samples since start)
        '''
        ret = self.avg_loss_since_start(), self.avg_metric_since_start(), self.samples_since_start
        self.loss_since_start    = 0
        self.metric_since_start  = 0
        self.samples_since_start = 0
        self.updates_since_start = 0
        return ret

    def reset_last(self):
        '''
        Resets the 'last' accumulators

        Returns: tuple of (average loss since last, average metric since last, samples since last)
        '''
        ret = self.avg_loss_since_last(), self.avg_metric_since_last(), self.samples_since_last
        self.loss_since_last    = 0
        self.metric_since_last  = 0
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
        If on an arithmetic schedule print an epoch summary using the 'start' accumulators.
        If on a geometric schedule does nothing.

        Args:
            with_metric (`bool`): if `False` it only prints the loss, otherwise it prints both the loss and the metric
        '''
        self.epochs += 1
        if self.freq > 0:
            epoch_end_time = time.time()
            time_delta = epoch_end_time - self.epoch_start_time
            speed = 0
            avg_loss, avg_metric, samples = (0, 0, 0)
            if self.samples_since_start != 0:
                avg_loss, avg_metric, samples = self.reset_start()
            if (time_delta > 0):
                speed = samples / time_delta
                self.epoch_start_time = epoch_end_time
            if with_metric:
                self.___logprint("Finished Epoch[{} of {}]: {}loss = {:0.6f} * {}, metric = {:0.1f}% * {} {:0.3f}s ({:5.1f} samples per second)".format(self.epochs, self.num_epochs, self.tag, avg_loss, samples, avg_metric*100.0, samples, time_delta, speed))
            else:
                self.___logprint("Finished Epoch[{} of {}]: {}loss = {:0.6f} * {} {:0.3f}s ({:5.1f} samples per second)".format(self.epochs, self.num_epochs, self.tag, avg_loss, samples, time_delta, speed))

            # For logging to TensorBoard, we use self.total_updates as it does not reset after each epoch.
            self.update_value('epoch_avg_loss', avg_loss, self.epochs)
            if with_metric:
                self.update_value('epoch_avg_metric', avg_metric * 100.0, self.epochs)

            return avg_loss, avg_metric, samples  # BUGBUG: for freq=0, we don't return anything here

    def ___gererate_progress_heartbeat(self):
        timer_delta = time.time() - self.progress_timer_time
        
        # print progress no sooner than 10s apart
        if timer_delta > 10 and self.gen_heartbeat:
            # print to stdout
            print("PROGRESS: 0.00%")
            self.progress_timer_time = time.time()

    def update(self, loss, minibatch_size, metric=None):
        '''
        Updates the accumulators using the loss, the minibatch_size and the optional metric.

        Args:
            loss (`float`): the value with which to update the loss accumulators
            minibatch_size (`int`): the value with which to update the samples accumulator
            metric (`float` or `None`): if `None` do not update the metric
             accumulators, otherwise update with the given value
        '''
        self.samples_since_start += minibatch_size
        self.samples_since_last  += minibatch_size
        self.loss_since_start    += loss * minibatch_size
        self.loss_since_last     += loss * minibatch_size
        self.updates_since_start += 1
        self.total_updates       += 1

        if metric is not None:
            self.metric_since_start += metric * minibatch_size
            self.metric_since_last  += metric * minibatch_size

        if self.epoch_start_time == 0:
            self.epoch_start_time = time.time()

        self.___gererate_progress_heartbeat()

        if self.freq == 0 and (self.updates_since_start+1) & self.updates_since_start == 0:
            avg_loss, avg_metric, samples = self.reset_last()
            if metric is not None:
                self.___logprint(' {:8.3g}   {:8.3g}   {:8.3g}   {:8.3g}    {:10d}'.format(
                    self.avg_loss_since_start(), avg_loss,
                    self.avg_metric_since_start(), avg_metric,
                    self.samples_since_start))
            else:
                self.___logprint(' {:8.3g}   {:8.3g}   {:8s}   {:8s}    {:10d}'.format(
                    self.avg_loss_since_start(), avg_loss,
                    '', '', self.samples_since_start))
        elif self.freq > 0 and (self.updates_since_start % self.freq == 0 or self.updates_since_start <= self.first):
            avg_loss, avg_metric, samples = self.reset_last()

            if self.updates_since_start <= self.first: # printing individual MBs
                first_mb = self.updates_since_start
            else:
                first_mb = max(self.updates_since_start - self.freq + 1, self.first+1)

            if metric is not None:
                self.___logprint(' Minibatch[{:4d}-{:4d}]: loss = {:0.6f} * {:d}, metric = {:0.1f}% * {:d}'.format(
                    first_mb, self.updates_since_start, avg_loss, samples, avg_metric*100.0, samples))
            else:
                self.___logprint(' Minibatch[{:4d}-{:4d}]: loss = {:0.6f} * {:d}'.format(
                    first_mb, self.updates_since_start, avg_loss, samples))

            if self.updates_since_start > self.first:
                # For logging to TensorBoard, we use self.total_updates as it does not reset after each epoch.
                self.update_value('mb_avg_loss', avg_loss, self.total_updates)
                if metric is not None:
                    self.update_value('mb_avg_metric', avg_metric * 100.0, self.total_updates)

    def update_with_trainer(self, trainer, with_metric=False):
        '''
        Updates the accumulators using the loss, the minibatch_size and optionally the metric
        using the information from the ``trainer``.

        Args:
            trainer (:class:`cntk.trainer.Trainer`): trainer from which information is gathered
            with_metric (`bool`): whether to update the metric accumulators
        '''
        if trainer.previous_minibatch_sample_count == 0:
            return
        self.update(
            trainer.previous_minibatch_loss_average,
            trainer.previous_minibatch_sample_count, 
            trainer.previous_minibatch_evaluation_average if with_metric else None)

    def update_value(self, name, value, step):
        '''
        Updates a named value at the given step.

        Args:
            name (string): name of the value to update.
            value (float): the updated value.
            step (int): step at which the value is recorded.
        '''
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.write_value(str(name), float(value), int(step))


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
