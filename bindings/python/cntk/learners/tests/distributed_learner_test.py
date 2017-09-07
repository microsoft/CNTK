import numpy as np
import os
import sys
import signal
import subprocess
import time
import re
import pytest
import argparse
import cntk as C

TIMEOUT_SECONDS = 300

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

class SimpleTrainer:
    def __init__(self):
        self.input_dim = 40000
        self.embed_dim = 100
        self.batch_size = 20
        i = C.input_variable((self.input_dim,), is_sparse=True)
        self.p = C.parameter(shape=(self.input_dim, self.embed_dim), init=1)
        o = C.times(i, self.p)
        z = C.reduce_sum(o)
        learner = C.data_parallel_distributed_learner(C.sgd(z.parameters, C.learning_rate_schedule(0.01, unit=C.learners.UnitType.sample)))
        self.trainer = C.Trainer(z, (z, None), learner, [])

    def train_minibatch(self, input_indices):
        data = C.Value.one_hot(input_indices, num_classes=self.input_dim)
        self.trainer.train_minibatch(data)

def data_parallel_sgd_on_sparse(outdir, gpu):
    if gpu:
        # test with only one GPU
        C.try_set_default_device(C.gpu(0))
    else:
        # CPU sparse aggregation is not implemented, so turn it off
        # note we only need to explicitly do this when running with CPU device on a GPU build
        # For CPU build it's disabled by default
        C.cntk_py.use_sparse_gradient_aggregation_in_data_parallel_sgd(False)

    trainer = SimpleTrainer()
    np.random.seed(C.Communicator.rank())
    indices = (np.random.random((trainer.batch_size,))*(trainer.input_dim-1)).astype(np.int)
    trainer.train_minibatch(indices)
    np.save(os.path.join(outdir, str(C.Communicator.rank())), trainer.p.value)

# test entrance
def test_data_parallel_sgd_on_sparse(tmpdir, device_id):
    launch_args = ['--outputdir', str(tmpdir), '--mode', 'data_parallel_sgd_on_sparse']
    if device_id >= 0:
        launch_args += ['--gpu']
    mpiexec_execute(__file__, ['-n', '2'], launch_args)
    
    p0 = np.load(os.path.join(str(tmpdir), '0.npy'))
    p1 = np.load(os.path.join(str(tmpdir), '1.npy'))
    assert np.allclose(p0, p1)
    
    ref_trainer = SimpleTrainer()
    np.random.seed(0)
    indices0 = (np.random.random((ref_trainer.batch_size,))*(ref_trainer.input_dim-1)).astype(np.int)
    np.random.seed(1)
    indices1 = (np.random.random((ref_trainer.batch_size,))*(ref_trainer.input_dim-1)).astype(np.int)
    ref_trainer.train_minibatch(np.concatenate([indices0, indices1]))
    assert np.allclose(p0, ref_trainer.p.value)

#mpiexec entrance
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-outputdir', '--outputdir')
    parser.add_argument('-mode', '--mode')
    parser.add_argument('-gpu', '--gpu', action='store_true')

    args = vars(parser.parse_args())
    
    if (args['mode'] == 'data_parallel_sgd_on_sparse'):
        data_parallel_sgd_on_sparse(args['outputdir'], args['gpu'])
    else:
        raise Exception('Unsupported mode ' + args['mode'])
        
    C.Communicator.finalize()