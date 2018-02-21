import argparse
import collections
import os
import pickle
import pytest
import re
import signal
import subprocess
import sys
import time
import numpy as np
import cntk as C

TIMEOUT_SECONDS = 300
NUM_WORKERS = 4
NUM_BATCHES = 10
BATCH_SIZE_PER_WORKER = 20

def mpiexec_execute(script, mpiexec_params, params, timeout_seconds=TIMEOUT_SECONDS):
    cmd = ['mpiexec'] + mpiexec_params + ['python', script] + params
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    if sys.version_info[0] < 3:
        out = p.communicate()[0]
    else:
        try:
            out = p.communicate(timeout=timeout_seconds)[0]  # in case we have a hang
        except subprocess.TimeoutExpired:
            os.kill(p.pid, signal.CTRL_C_EVENT)
            raise RuntimeError('Timeout in mpiexec, possibly hang')
    str_out = out.decode(sys.getdefaultencoding())
    return str_out

BlockMomentumConfig = collections.namedtuple('BlockMomentumConfig', 'block_momentum_as_time_constant block_learning_rate block_size distributed_after')
DataParallelConfig = collections.namedtuple('DataParallelConfig', 'num_quantization_bits distributed_after')
    
class SimpleTrainer:
    def __init__(self, mode, config):
        self.create_model()
        self.create_trainer(mode, config)
        
    def create_model(self):
        self.input_dim = 1000
        self.embed_dim = 30
        i = C.input_variable((self.input_dim,), is_sparse=True)
        self.p = C.parameter(shape=(self.input_dim, self.embed_dim), init=1)
        o = C.times(i, self.p)
        self.z = C.reduce_sum(o)

    def create_trainer(self, mode, config):
        learner = self.create_distributed_learner(mode, config)
        self.trainer = C.Trainer(self.z, (self.z, None), learner, []) if learner else None

    def create_distributed_learner(self, mode, config):
        local_learner = C.sgd(self.z.parameters, C.learning_parameter_schedule_per_sample(0.01))
        try:
            if mode == 'data_parallel':
                if config is None:
                    config = DataParallelConfig(num_quantization_bits=32, distributed_after=0)
                learner = C.data_parallel_distributed_learner(local_learner, num_quantization_bits=config.num_quantization_bits, distributed_after=config.distributed_after)
            elif mode == 'block_momentum':
                if config is None:
                    # the default config to match data parallel SGD
                    config = BlockMomentumConfig(block_momentum_as_time_constant=0, block_learning_rate=1, block_size=NUM_WORKERS, distributed_after=0)
                learner = C.block_momentum_distributed_learner(local_learner, block_momentum_as_time_constant=config.block_momentum_as_time_constant, block_learning_rate=config.block_learning_rate, block_size=config.block_size, distributed_after=config.distributed_after)
            else:
                learner = local_learner
        except RuntimeError:
            learner = None
        return learner

    def train_minibatch(self, input_indices):
        data = C.Value.one_hot(input_indices, num_classes=self.input_dim)
        self.trainer.train_minibatch(data)

def set_np_random_seed(rank, batch):
    np.random.seed(rank + 10 * batch)
        
def distributed_worker(outdir, gpu, mode, config):
    if gpu:
        # test with only one GPU
        C.try_set_default_device(C.gpu(0))
    else:
        # CPU sparse aggregation is not implemented, so turn it off
        # note we only need to explicitly do this when running with CPU device on a GPU build
        # For CPU build it's disabled by default
        C.cntk_py.use_sparse_gradient_aggregation_in_data_parallel_sgd(False)

    trainer = SimpleTrainer(mode, config)
    for batch in range(NUM_BATCHES):
        set_np_random_seed(C.Communicator.rank(), batch)
        indices = (np.random.random((BATCH_SIZE_PER_WORKER,))*(trainer.input_dim-1)).astype(np.int)
        trainer.train_minibatch(indices)
        checkpoint_file = os.path.join(outdir, mode+str(batch))
        trainer.trainer.save_checkpoint(checkpoint_file)
        trainer.trainer.restore_from_checkpoint(checkpoint_file)
    
    # save a checkpoint to force sync after last minibatch
    trainer.trainer.save_checkpoint(os.path.join(outdir, mode+'_last'))
    np.save(os.path.join(outdir, mode+str(C.Communicator.rank())), trainer.p.value)

TRAINING_SETTINGS = [
    ('data_parallel', None),
    ('block_momentum', None),
    ('block_momentum', BlockMomentumConfig(block_momentum_as_time_constant=4000, block_learning_rate=2, block_size=NUM_WORKERS*BATCH_SIZE_PER_WORKER*3, distributed_after=NUM_WORKERS*BATCH_SIZE_PER_WORKER*2)),
    ('data_parallel', DataParallelConfig(num_quantization_bits=1, distributed_after=NUM_WORKERS*BATCH_SIZE_PER_WORKER*2)),
]

@pytest.mark.parametrize("mode, config", TRAINING_SETTINGS)
def test_distributed_training_accuracy(tmpdir, device_id, mode, config):
    ref_trainer = SimpleTrainer(None, None)

    # test if mode is available
    if not ref_trainer.create_distributed_learner(mode, config):
        pytest.skip("unsupported distributed learner mode")

    # run distributed training and check if all workers get the same model
    launch_args = ['--outputdir', str(tmpdir), '--mode', mode]
    
    if config:
        config_filename = os.path.join(str(tmpdir),'config.pkl')
        with open(config_filename, 'wb') as pkl:
            pickle.dump(config, pkl)
        launch_args += ['--config', config_filename]
    
    if device_id >= 0:
        launch_args += ['--gpu']

    mpiexec_execute(__file__, ['-n', str(NUM_WORKERS)], launch_args)

    p0 = np.load(os.path.join(str(tmpdir), mode+'0.npy'))
    for rank in range(NUM_WORKERS):
        p = np.load(os.path.join(str(tmpdir), mode+str(rank)+'.npy'))
        assert np.allclose(p0, p)
    
    # only compares with single worker with default config
    if config is not None:
        return

    # reference training on single worker, by concatenating data on all workers
    for batch in range(NUM_BATCHES):
        indices = None
        for rank in range(NUM_WORKERS):
            set_np_random_seed(rank, batch)
            rank_indices = (np.random.random((BATCH_SIZE_PER_WORKER,))*(ref_trainer.input_dim-1)).astype(np.int)
            indices = np.concatenate([indices, rank_indices]) if indices is not None else rank_indices
        ref_trainer.train_minibatch(indices)

    assert np.allclose(p0, ref_trainer.p.value)

#mpiexec entrance
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-outputdir', '--outputdir')
    parser.add_argument('-mode', '--mode')
    parser.add_argument('-gpu', '--gpu', action='store_true')
    parser.add_argument('-config', '--config', required=False, default=None)
    args = vars(parser.parse_args())
    
    config = None
    if args['config'] is not None:
        with open(args['config'], 'rb') as pkl:
            config = pickle.load(pkl)

    distributed_worker(args['outputdir'], args['gpu'], args['mode'], config)
    C.Communicator.finalize()