"""
This test extends the bmuf_metrics_aggregation_test and tests multiple learners in the distributed training.
"""
import pytest
import cntk
import numpy as np
import sys, os
sys.path.append(os.path.dirname(__file__))
from bmuf_metrics_aggregation_test import SimpleBMUFTrainer
from bmuf_metrics_aggregation_test import get_minibatch
from bmuf_metrics_aggregation_test import mpi_worker
from distributed_learner_test import mpiexec_execute

import argparse
import re
import platform

cntk.cntk_py.set_fixed_random_seed(1)
#cntk.logging.set_trace_level(cntk.logging.TraceLevel.Info)

feat_dim = 5
label_dim = 3
cell_dim = 5
seq_len = 20
num_batches = 101
batch_size = 10
progress_freq =10
NUM_WORKERS = 4

class MultiLearnerMUFTrainer(SimpleBMUFTrainer):
    def __init__(self, frame_mode=False):
        SimpleBMUFTrainer.__init__(self, frame_mode)

    def create_trainer(self):
        try:
            p = self.output.parameters
            # Three of four parameters are learned by block_momentum_distributed_learner.
            bmd_learner = cntk.block_momentum_distributed_learner(cntk.momentum_sgd([p[0],p[1],p[2]], cntk.learning_parameter_schedule(0.0001), cntk.momentum_as_time_constant_schedule(1000)), 
                                                                    block_size=1000, block_learning_rate=0.01, block_momentum_as_time_constant=1000)

            # New API to mark which learner is to use for metric aggregaion.
            bmd_learner.set_as_metric_aggregator()

            # The last parameter is learned by the data_parallel_distributed_learner.
            momentum_schedule = cntk.momentum_schedule_per_sample(0.9990913221888589)
            lr_per_sample = cntk.learning_parameter_schedule_per_sample(0.007)
            dpd_learner = cntk.data_parallel_distributed_learner(cntk.momentum_sgd([p[3]], lr_per_sample, momentum_schedule, True))

            comm_rank = cntk.distributed.Communicator.rank()
            self.trainer = cntk.Trainer(self.output, (self.ce, self.err), [bmd_learner, dpd_learner], [cntk.logging.ProgressPrinter(freq=progress_freq, tag="Training", rank=comm_rank)])
        except RuntimeError:
            self.trainer = None
        return


def mpi_worker_multi_learner(working_dir, checkpoint_dir, mb_source, gpu):
    comm_rank = cntk.distributed.Communicator.rank()
    np.random.seed(comm_rank)
    
    if gpu:
        # test with only one GPU
        cntk.try_set_default_device(cntk.gpu(0))

    frame_mode = (mb_source == "ctf_frame")
    bmuf = MultiLearnerMUFTrainer(frame_mode)

    checkpoint_performed = False
    for i, data in enumerate(get_minibatch(bmuf, working_dir, mb_source)):
        bmuf.trainer.train_minibatch(data)
        if i % 50 == 0:
            bmuf.trainer.summarize_training_progress()
            if not checkpoint_performed and not checkpoint_dir == "":
                bmuf.trainer.save_checkpoint(checkpoint_dir)
                bmuf.trainer.restore_from_checkpoint(checkpoint_dir)
                checkpoint_performed = True


#MB_SOURCES = ["numpy"]
MB_SOURCES = ["ctf_utterance"]
@pytest.mark.parametrize("mb_source", MB_SOURCES)
def test_multi_learner_bmuf_correct_metrics_averaging(tmpdir, device_id, mb_source):
    if platform.system() == 'Linux':
        pytest.skip('test only runs on Windows due to mpiexec -l option')

    # check whether trainer can be initialized or not
    bmuf = MultiLearnerMUFTrainer()
    if not bmuf.trainer:
        pytest.skip('BMUF not available on this build')

    launch_args = []
    if device_id >= 0:
        launch_args += ['--gpu']

    launch_args += ["--outputdir", str(tmpdir)]
    launch_args += ["--mb_source", mb_source]

    ret_str = mpiexec_execute(__file__, ['-n', str(NUM_WORKERS), '-l'], launch_args)
    #print(ret_str)

    # [0]Finished Epoch[1]: [Training] loss = 1.663636 * 10, metric = 52.40% * 10 0.890s ( 11.2 samples/s);
    regex_pattern = r"\[(?P<worker_rank>\d)\].*? Epoch\[(?P<epoch>\d+)\].*? loss = (?P<loss>\d+\.\d+) .*? metric = (?P<metric>\d+\.\d+)"
    loss_perepoch_perworker = {i:{} for i in range(NUM_WORKERS)}
    for match in re.finditer(regex_pattern, ret_str):
        rank = int(match.groupdict()["worker_rank"])
        epoch = int(match.groupdict()["epoch"])
        loss = match.groupdict()["loss"]
        metric = match.groupdict()["metric"]
        loss_perepoch_perworker[rank].update({epoch:(loss, metric)})

    num_epochs_per_worker = list(map(len,loss_perepoch_perworker.values()))

    #assert that data exists
    assert len(num_epochs_per_worker) != 0

    #assert that number of epochs isn't zero for 1st worker.
    assert num_epochs_per_worker[0] != 0
    
    # assert all workers have same number of epochs
    assert min(num_epochs_per_worker) == max(num_epochs_per_worker)
    
    # assert all workers have same loss and metric values
    loss_per_worker = loss_perepoch_perworker.values()
    loss_per_worker_epochsort = []
    for epoch_losses in loss_per_worker:
        loss_per_worker_epochsort.append([epoch_losses[i] for i in sorted(epoch_losses)])
        
    assert all([loss_per_worker_epochsort[0] == i for i in loss_per_worker_epochsort])

    # Do the same test with checkpoint and compare the results.
    launch_args += ["--checkpointdir", str(tmpdir.join('checkpoint'))]

    ret_str = mpiexec_execute(__file__, ['-n', str(NUM_WORKERS), '-l'], launch_args)
    #print(ret_str)    

    loss_perepoch_perworker = {i:{} for i in range(NUM_WORKERS)}
    for match in re.finditer(regex_pattern, ret_str):
        rank = int(match.groupdict()["worker_rank"])
        epoch = int(match.groupdict()["epoch"])
        loss = match.groupdict()["loss"]
        metric = match.groupdict()["metric"]
        loss_perepoch_perworker[rank].update({epoch:(loss, metric)})

    num_epochs_per_worker = list(map(len,loss_perepoch_perworker.values()))

    #assert that data exists
    assert len(num_epochs_per_worker) != 0

    #assert that number of epochs isn't zero for 1st worker.
    assert num_epochs_per_worker[0] != 0

    # assert all workers have same number of epochs
    assert min(num_epochs_per_worker) == max(num_epochs_per_worker)

    # assert all workers have same loss and metric values
    loss_per_worker = loss_perepoch_perworker.values()
    multi_learner_loss_per_worker_epochsort = []
    for epoch_losses in loss_per_worker:
        multi_learner_loss_per_worker_epochsort.append([epoch_losses[i] for i in sorted(epoch_losses)])

    # Compare no checkpoint loss values to checkpoint loss values.
    assert all([loss_per_worker_epochsort[0] == i for i in multi_learner_loss_per_worker_epochsort])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-outputdir', '--outputdir')
    parser.add_argument('-checkpointdir', '--checkpointdir')
    parser.add_argument('-mb_source', '--mb_source')
    parser.add_argument('-gpu', '--gpu', action='store_true')
    args = vars(parser.parse_args())

    if args["checkpointdir"]:
        mpi_worker_multi_learner(args["outputdir"], args["checkpointdir"], args["mb_source"], args["gpu"])
    else:
        mpi_worker_multi_learner(args["outputdir"], "", args["mb_source"], args["gpu"])
    cntk.distributed.Communicator.finalize()
