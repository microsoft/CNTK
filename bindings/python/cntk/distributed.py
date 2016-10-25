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
    def __init__(self, distributed_communicator):
        self.data = distributed_communicator
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

    def barrier(self):
        '''
        sync point to make sure all workers reach the same state
        '''
        self.data.barrier()
        return
        
    @staticmethod
    def finalize():
        cntk_py.DistributedCommunicator.finalize();
        return

class distributed_trainer:
    '''
    A distributed trainer that can be passed to the :class:`cntk.trainer.Trainer`

    Args:
       trainer (:class:`cntk.cntk_py.DistributedTrainer`): internal distributed trainer
    '''
    def __init__(self, distributed_trainer):
        self.data = distributed_trainer
        
def mpi_communicator():
    '''
    Creates a mpi communicator

    Returns:
        :class:`cntk.cntk_py.DistributedCommunicator`: a distributed communicator
    '''
    return cntk_py.mpicommunicator()

def quantized_mpi_communicator(num_quantization_bits):
    '''
    Creates a quantized mpi communicator

    Args:
        num_quantization_bits (`int`): num_quantization_bits

    Returns:
        :class:`cntk.cntk_py.QuantizedDistributedCommunicator`: a quantized distributed communicator
    '''
    return cntk_py.quantized_mpicommunicator(True, True, num_quantization_bits)

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
    return distributed_trainer(cntk_py.create_data_parallel_distributed_trainer(communicator.data, use_async_buffered_parameter_update))