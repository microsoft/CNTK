# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from . import cntk_py
from . import trainer
from .utils import typemap

# Preload libmpi.so.12 for non-Windows platform to work around MPI_Init failure bug
# https://xrunhprof.wordpress.com/2014/11/04/an-openmpi-python-and-dlopen-issue/
# If other OS has similar OpenMPI MPI_Init failure, add dll load to global here
import platform
import ctypes
if platform.system() == 'Linux':
    ctypes.CDLL("libmpi.so.12", mode=ctypes.RTLD_GLOBAL)

__doc__= '''\
Distributed learners manage learners in distributed environment.
'''

class WorkerDescriptor(cntk_py.DistributedWorkerDescriptor):
    '''
    Distributed worker descriptor, returned by :class:`Communicator` instance.
    '''

    @property
    def global_rank(self):
        '''
        The global rank of the worker.
        '''
        return super(WorkerDescriptor, self).m_global_rank

    @property
    def host_id(self):
        '''
        The host id of the worker.
        '''
        return super(WorkerDescriptor, self).m_host_id

class Communicator(cntk_py.DistributedCommunicator):
    '''
    A communicator interface exposing communication primitives that serve as building blocks 
    for distributed training.
    '''

    @typemap
    def workers(self):
        '''
        Returns workers in this communicator.
        
        Returns:
            (`list`) of :class:`WorkerDescriptor`: workers in this communicator.
        '''
        return super(Communicator, self).workers()

    @typemap
    def current_worker(self):
        '''
        Returns worker descriptor of current process.
        
        Returns:
            :class:`WorkerDescriptor`: descriptor of current process.
        '''
        return super(Communicator, self).current_worker()

    def barrier(self):
        '''
        Sync point to make sure all workers reach the same state.
        '''
        super(Communicator, self).barrier()

    def is_main(self):
        '''
        Indicates if the current communicator is instantiated on the main node. The node with rank 0 is considered the main.
        '''
        return super(Communicator, self).current_worker().is_main()

    @staticmethod
    def finalize():
        '''
        Should be called when all communication is finished. No more communication should happen after this call.
        '''
        cntk_py.DistributedCommunicator.finalize()

    @staticmethod
    def num_workers():
        '''
        Returns information about all MPI workers.
        '''
        return cntk_py.number_of_workers()

    @staticmethod
    def rank():
        '''
        Returns rank of current process.
        '''
        return cntk_py.worker_global_rank()

class DistributedLearner(cntk_py.DistributedLearner):
    '''
    A distributed learner that handles data like gradients/momentums across multiple MPI workers
    '''
    
    @typemap
    def communicator(self):
        '''
        Returns the distributed communicator that talks to other MPI workers
        
        Returns:
            :class:`Communicator`: descriptor of current process.
        '''
        return super(DistributedLearner, self).get_communicator()

@typemap
def data_parallel_distributed_learner(learner, distributed_after=0, num_quantization_bits=32, use_async_buffered_parameter_update=False):
    '''
    Creates a data parallel distributed learner

    Args:
        learner: a local learner (i.e. sgd)
        distributed_after (int): number of samples after which distributed training starts
        num_quantization_bits (int): number of bits for quantization (1 to 32)
        use_async_buffered_parameter_update (bool): use async buffered parameter update
    Returns:
        a distributed learner instance
    '''
    if (num_quantization_bits < 32):
        return cntk_py.create_quantized_data_parallel_distributed_learner(
            cntk_py.quantized_mpicommunicator(True, True, num_quantization_bits),
            learner,
            distributed_after,
            use_async_buffered_parameter_update)
    else:
        return cntk_py.create_data_parallel_distributed_learner(
            cntk_py.mpicommunicator(),
            learner,
            distributed_after,
            use_async_buffered_parameter_update)

@typemap
def block_momentum_distributed_learner(learner, block_size, block_momentum_as_time_constant=None, use_nestrov_momentum=True, reset_sgd_momentum_after_aggregation=True, block_learning_rate=1.0, distributed_after=0):
    '''
    Creates a block momentum distributed learner. See [1] for more
    information.

    Block Momentum divides the full dataset into M non-overlapping blocks,
    and each block is partitioned into N non-overlapping splits.

    During training, a random, unprocessed block is randomly taken by the trainer
    and the N partitions of this block are dispatched on the workers.

    Args:
        learner: a local learner (i.e. sgd)
        block_size (int): size of the partition in samples
        block_momentum_as_time_constant (float): block momentum as time constant
        use_nestrov_momentum (bool): use nestrov momentum
        reset_sgd_momentum_after_aggregation (bool): reset SGD momentum after aggregation
        block_learning_rate (float): block learning rate
        distributed_after (int): number of samples after which distributed training starts

    Returns:
        a distributed learner instance

    See also:
        [1] K. Chen and Q. Huo. `Scalable training of deep learning machines
        by incremental block training with intra-block parallel optimization
        and blockwise model-update filtering
        <https://www.microsoft.com/en-us/research/wp-content/uploads/2016/08/0005880.pdf>`_. 
        Proceedings of ICASSP, 2016. 
    '''
    if block_momentum_as_time_constant == None:
        return cntk_py.create_block_momentum_distributed_learner(
            cntk_py.mpicommunicator(),
            learner,
            distributed_after,
            block_size,
            use_nestrov_momentum,
            reset_sgd_momentum_after_aggregation,
            block_learning_rate)
    else:
        return cntk_py.create_block_momentum_distributed_learner(
            cntk_py.mpicommunicator(),
            learner,
            distributed_after,
            block_size,
            block_momentum_as_time_constant,
            use_nestrov_momentum,
            reset_sgd_momentum_after_aggregation,
            block_learning_rate)

