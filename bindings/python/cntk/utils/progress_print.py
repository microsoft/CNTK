# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

# TODO: Let's switch to import logging in the future instead of print. [ebarsoum]
class ProgressPrinter:
    '''
    Accumulates training time statistics (loss and metric)
    and pretty prints them as training progresses.

    It provides the number of samples, average loss and average metric
    since the last print or since the start of accumulation.
    '''
    def __init__(self, freq=None, first=0, tag=''):
        '''
        Constructor. The optional ``freq`` parameter determines how often
        printing will occur. The value of 0 means an geometric
        schedule (1,2,4,...). A value > 0 means a arithmetic schedule
        (freq, 2*freq, 3*freq,...), and a value of None means no per-minibatch log.
        '''
        from sys import maxsize
        if freq is None:
            freq = maxsize
        self.loss_since_start    = 0
        self.metric_since_start  = 0
        self.samples_since_start = 0
        self.loss_since_last     = 0
        self.metric_since_last   = 0
        self.samples_since_last  = 0
        self.updates             = 0
        self.epochs              = 0
        self.freq                = freq
        self.first               = first
        self.tag                 = '' if not tag else "[{}] ".format(tag)

        if freq==0:
            print(' average      since    average      since      examples')
            print('    loss       last     metric       last              ')
            print(' ------------------------------------------------------')

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

    def epoch_summary(self, with_metric=False):
        '''
        If on an arithmetic schedule print an epoch summary using the 'start' accumulators.
        If on a geometric schedule does nothing.

        Args:
            with_metric (`bool`): if `False` it only prints the loss, otherwise it prints both the loss and the metric
        '''
        self.epochs += 1
        if self.freq > 0:
            self.updates = 0
            avg_loss, avg_metric, samples = self.reset_start()
            if with_metric:
                print("Finished Epoch [{}]: {}loss = {:0.6f} * {}, metric = {:0.1f}% * {}".format(self.epochs, self.tag, avg_loss, samples, avg_metric*100.0, samples))
            else:
                print("Finished Epoch [{}]: {}loss = {:0.6f} * {}".format(self.epochs, self.tag, avg_loss, samples))
            return avg_loss, avg_metric, samples  # BUGBUG: for freq=0, we don't return anything here

    def update(self, loss, minibatch_size, metric=None):
        '''
        Updates the accumulators using the loss, the minibatch_size and the optional metric.

        Args:
            loss (`float`): the value with which to update the loss accumulators
            minibatch_size (`int`): the value with which to update the samples accumulator
            metric (`float` or `None`): if `None` do not update the metric
             accumulators, otherwise update with the given value
        '''
        self.updates             += 1
        self.samples_since_start += minibatch_size
        self.samples_since_last  += minibatch_size
        self.loss_since_start    += loss * minibatch_size
        self.loss_since_last     += loss * minibatch_size
        if metric is not None:
            self.metric_since_start += metric * minibatch_size
            self.metric_since_last  += metric * minibatch_size
        if self.freq == 0 and (self.updates+1) & self.updates == 0:
            avg_loss, avg_metric, samples = self.reset_last()
            if metric is not None:
                print(' {:8.3g}   {:8.3g}   {:8.3g}   {:8.3g}    {:10d}'.format(
                    self.avg_loss_since_start(), avg_loss,
                    self.avg_metric_since_start(), avg_metric,
                    self.samples_since_start))
            else:
                print(' {:8.3g}   {:8.3g}   {:8s}   {:8s}    {:10d}'.format(
                    self.avg_loss_since_start(), avg_loss,
                    '', '', self.samples_since_start))
        elif self.freq > 0 and (self.updates % self.freq == 0 or self.updates <= self.first):
            avg_loss, avg_metric, samples = self.reset_last()
            if self.updates <= self.first: # printing individual MBs
                first_mb = self.updates
            else:
                first_mb = max(self.updates - self.freq + 1, self.first+1)
            if metric is not None:
                print(' Minibatch[{:4d}-{:4d}]: loss = {:0.6f} * {:d}, metric = {:0.1f}% * {:d}'.format(
                    first_mb, self.updates, avg_loss, samples, avg_metric*100.0, samples))
            else:
                print(' Minibatch[{:4d}-{:4d}]: loss = {:0.6f} * {:d}'.format(
                    first_mb, self.updates, avg_loss, samples))

    def update_with_trainer(self, trainer, with_metric=False):
        '''
        Updates the accumulators using the loss, the minibatch_size and optionally the metric
        using the information from the ``trainer``.

        Args:
            trainer (:class:`cntk.trainer.Trainer`): trainer from which information is gathered
            with_metric (`bool`): whether to update the metric accumulators
        '''
        self.update(trainer.previous_minibatch_loss_average,trainer.previous_minibatch_sample_count, trainer.previous_minibatch_evaluation_average if with_metric else None)

        
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
            print ("\t{}".format(p.shape))
