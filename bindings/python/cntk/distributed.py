# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from . import cntk_py
from . import trainer

__doc__= '''\
Distributed trainers manages trainers in distributed environment.
'''

class worker_descriptor:
    '''
    Distributed worker descriptor, returned by :class:`cntk.distributed.communicator` instance.

    Args:
       descriptor (:class:`cntk.cntk_py.DistributedWorkerDescriptor`): internal distributed worker descriptor
    '''
    def __init__(self, descriptor):
        self.data = descriptor
        return
    
    @property
    def global_rank(self):
        '''
        Returns the global rank of the worker.

        Returns:
            `int`: the global rank of the worker.
        '''
        return self.data.m_global_rank

    @property
    def host_id(self):
        '''
        Returns the host id of the worker.

        Returns:
            `str`: the host id of the worker.
        '''
        return self.data.m_host_id

class communicator:
    '''
    A communicator interface exposing communication primitives that serve as building blocks 
    for distributed training.
    '''
    def __init__(self):
        self.data = cntk_py.mpicommunicator()
        return
        
    def workers(self):
        '''
        Returns workers in this communicator.
        
        Returns:
            (`list`) of :class:`cntk.distributed.worker_descriptor`: workers in this communicator.
        '''
        raw_list = self.data.workers()
        ret = []
        for w in raw_list:
            ret.append(worker_descriptor(w))
        return ret

    def current_worker(self):
        '''
        Returns worker descriptor of current process.
        
        Returns:
            :class:`cntk.distributed.worker_descriptor`: descriptor of current process.
        '''
        raw = self.data.current_worker()
        return worker_descriptor(raw)

    def sub_group(self, workers):
        '''
        Creates a new distributed communicator comprising of a subset of the workers in this communicator.
        
        Args:
            workers (`list`): list of :class:`cntk.distributed.worker_descriptor` of workers in the new communicator

        Returns:
            :class:`cntk.distributed.communicator`: comprising specified workers
        '''
        raw_list = []
        for w in workers:
            raw_list.append(w.data)
        self.data.sub_group(raw_list)
        return

class trainer:
    '''
    A distributed trainer that can be passed to the :class:`cntk.trainer.Trainer`

    Args:
       trainer (:class:`cntk.cntk_py.DistributedTrainer`): internal distributed trainer
    '''
    def __init__(self, distributed_trainer):
        self.data = distributed_trainer

    @staticmethod
    def data_parallel_distributed_trainer(communicator, use_async_buffered_parameter_update):
        '''
        Creates a data parallel distributed trainer using `communicator` with
        option `use_async_buffered_parameter_update`.

        Args:
            communicator (:class:`cntk.distributed.communicator`): distributed communicator
            use_async_buffered_parameter_update (`bool`): use async buffered parameter update

        Returns:
            :class:`cntk.distributed.trainer`: a distributed trainer instance
        '''
        return trainer(cntk_py.create_data_parallel_distributed_trainer(communicator.data, use_async_buffered_parameter_update))