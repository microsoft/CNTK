"""
This test extends the bmuf_metrics_aggregation_test and tests multiple learners in the distributed training.
"""
import pytest
import cntk
import numpy as np
import sys, os
import argparse
import re
import platform
sys.path.append(os.path.dirname(__file__))
cntk.cntk_py.set_fixed_random_seed(1)
from distributed_learner_test import mpiexec_execute
from bmuf_metrics_aggregation_test import get_minibatch

feat_dim = 5
label_dim = 3
cell_dim = 5
seq_len = 20
num_batches = 101
progress_freq =10

class SingleDataParallelTrainer():
    def __init__(self, frame_mode=False):
        self.create_model(frame_mode)
        self.create_trainer()

    def create_model(self, frame_mode=False):
        if frame_mode:
            self.feat = cntk.input_variable(shape=(feat_dim,))
            self.label = cntk.input_variable((label_dim,))

            net = cntk.layers.Sequential([cntk.layers.Dense(cell_dim), cntk.layers.Dense(label_dim)])
            self.output = net(self.feat)
        else:
            #sequence mode
            self.feat = cntk.sequence.input_variable(shape=(feat_dim,))
            self.label = cntk.sequence.input_variable((label_dim,))

            net = cntk.layers.Sequential([cntk.layers.Recurrence(cntk.layers.LSTM(shape=label_dim, cell_shape=(cell_dim,)))])
            self.output = net(self.feat)

        self.ce = cntk.cross_entropy_with_softmax(self.output, self.label)
        self.err = cntk.classification_error(self.output, self.label)

    def create_trainer(self):
        try:
            lr_per_sample = cntk.learning_parameter_schedule_per_sample(0.007)
            learner = cntk.data_parallel_distributed_learner(cntk.sgd(self.output.parameters, lr_per_sample))

            comm_rank = cntk.distributed.Communicator.rank()
            self.trainer = cntk.Trainer(self.output, (self.ce, self.err), [learner], [cntk.logging.ProgressPrinter(freq=progress_freq, tag="Training", rank=comm_rank)])
        except RuntimeError:
            self.trainer = None
        return

class TwoDataParallelTrainer(SingleDataParallelTrainer):
    def __init__(self, frame_mode=False):
        SingleDataParallelTrainer.__init__(self, frame_mode)

    def create_trainer(self):
        try:
            lr_per_sample = cntk.learning_parameter_schedule_per_sample(0.007)
            p = self.output.parameters
            # Three of four parameters are learned by first data_parallel_distributed_learner.
            learner1 = cntk.data_parallel_distributed_learner(cntk.sgd([p[0],p[1],p[2]], lr_per_sample))

            # New API to mark which learner is to use for metric aggregaion.
            learner1.set_as_metric_aggregator()

            # The last parameter is learned by another data_parallel_distributed_learner.
            learner2 = cntk.data_parallel_distributed_learner(cntk.sgd([p[3]], lr_per_sample))

            comm_rank = cntk.distributed.Communicator.rank()
            self.trainer = cntk.Trainer(self.output, (self.ce, self.err), [learner1, learner2], [cntk.logging.ProgressPrinter(freq=progress_freq, tag="Training", rank=comm_rank)])
        except RuntimeError:
            self.trainer = None
        return

class MultiLearnerTrainer(SingleDataParallelTrainer):
    def __init__(self, frame_mode=False):
        SingleDataParallelTrainer.__init__(self, frame_mode)

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

def mpi_worker_multi_learner(trainer, working_dir, checkpoint_dir, mb_source):
    comm_rank = cntk.distributed.Communicator.rank()
    np.random.seed(comm_rank)

    num_paritions = cntk.Communicator.num_workers();
    partition_index = cntk.Communicator.rank();
    checkpoint_performed = False
    for i, data in enumerate(get_minibatch(trainer, working_dir, mb_source, num_paritions, partition_index)):
        trainer.trainer.train_minibatch(data)
        if i % 50 == 0:
            trainer.trainer.summarize_training_progress()
            if not checkpoint_performed and not checkpoint_dir == "":
                trainer.trainer.save_checkpoint(checkpoint_dir)
                trainer.trainer.restore_from_checkpoint(checkpoint_dir)
                checkpoint_performed = True

def get_loss_perepoch_perworker(log_line, num_workers):
    # [0]Finished Epoch[1]: [Training] loss = 1.663636 * 10, metric = 52.40% * 10 0.890s ( 11.2 samples/s);
    regex_pattern = r"\[(?P<worker_rank>\d)\].*? Epoch\[(?P<epoch>\d+)\].*? loss = (?P<loss>\d+\.\d+) \* (?P<samples>\d+).*? metric = (?P<metric>\d+\.\d+)"
    loss_perepoch_perworker = {i:{} for i in range(num_workers)}
    for match in re.finditer(regex_pattern, log_line):
        rank = int(match.groupdict()["worker_rank"])
        epoch = int(match.groupdict()["epoch"])
        loss = match.groupdict()["loss"]
        metric = match.groupdict()["metric"]
        samples = int(match.groupdict()["samples"])
        loss_perepoch_perworker[rank].update({epoch:(loss, metric, samples)})
    return loss_perepoch_perworker

MB_SOURCES = ["ctf_frame"]
@pytest.mark.parametrize("mb_source", MB_SOURCES)
def test_single_data_parallel_learner_vs_two_data_parallel_learners(tmpdir, device_id, mb_source):
    if platform.system() == 'Linux':
        pytest.skip('test only runs on Windows due to mpiexec -l option')

    launch_args = []
    launch_args += ["--outputdir", str(tmpdir)]
    launch_args += ["--mb_source", mb_source]
    launch_args += ["--trainer_type", "single"]

    num_workers = 1 # use a single worker.
    ret_str = mpiexec_execute(__file__, ['-n', str(num_workers), '-l'], launch_args)
    print(ret_str)
    loss_perepoch_perworker = get_loss_perepoch_perworker(ret_str, num_workers)

    loss_per_worker = loss_perepoch_perworker.values()
    single_learner_loss_per_worker_epochsort = []
    for epoch_losses in loss_per_worker:
        single_learner_loss_per_worker_epochsort.append([epoch_losses[i] for i in sorted(epoch_losses)])

    launch_args = []
    launch_args += ["--outputdir", str(tmpdir)]
    launch_args += ["--mb_source", mb_source]
    launch_args += ["--trainer_type", "two"]

    num_workers = 2 # now run in distributed workers.
    ret_str = mpiexec_execute(__file__, ['-n', str(num_workers), '-l'], launch_args)
    print(ret_str)
    loss_perepoch_perworker = get_loss_perepoch_perworker(ret_str, num_workers)

    loss_per_worker = loss_perepoch_perworker.values()
    multi_learner_loss_per_worker_epochsort = []
    for epoch_losses in loss_per_worker:
        multi_learner_loss_per_worker_epochsort.append([epoch_losses[i] for i in sorted(epoch_losses)])

    assert all([single_learner_loss_per_worker_epochsort[0] == i for i in multi_learner_loss_per_worker_epochsort])

MB_SOURCES = ["ctf_frame"]
@pytest.mark.parametrize("mb_source", MB_SOURCES)
def test_multi_learner_bmuf_correct_metrics_averaging(tmpdir, device_id, mb_source):
    if platform.system() == 'Linux':
        pytest.skip('test only runs on Windows due to mpiexec -l option')

    num_workers = 2
    # check whether trainer can be initialized or not
    bmuf = MultiLearnerTrainer()
    if not bmuf.trainer:
        pytest.skip('BMUF not available on this build')

    launch_args = []
    launch_args += ["--outputdir", str(tmpdir)]
    launch_args += ["--mb_source", mb_source]
    launch_args += ["--trainer_type", "multi"]

    ret_str = mpiexec_execute(__file__, ['-n', str(num_workers), '-l'], launch_args)
    print(ret_str)
    loss_perepoch_perworker = get_loss_perepoch_perworker(ret_str, num_workers)

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

    ret_str = mpiexec_execute(__file__, ['-n', str(num_workers), '-l'], launch_args)
    print(ret_str)

    loss_perepoch_perworker = get_loss_perepoch_perworker(ret_str, num_workers)

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

    # Compare no checkpoint loss, matric, and num samples, to checkpoint loss values.
    for i in multi_learner_loss_per_worker_epochsort:
        for n in range(3):
            for m in range(3):
                assert np.allclose(float(loss_per_worker_epochsort[0][n][m]), float(i[n][m]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-outputdir', '--outputdir')
    parser.add_argument('-checkpointdir', '--checkpointdir')
    parser.add_argument('-mb_source', '--mb_source')
    parser.add_argument("-trainer_type","--trainer_type")
    args = vars(parser.parse_args())

    frame_mode = (args["mb_source"] == "ctf_frame")

    if args["trainer_type"] == "multi":
        trainer = MultiLearnerTrainer(frame_mode)
        if args["checkpointdir"]:
         
            mpi_worker_multi_learner(trainer, args["outputdir"], args["checkpointdir"], args["mb_source"])
        else:        
            mpi_worker_multi_learner(trainer, args["outputdir"], "", args["mb_source"])

    elif args["trainer_type"] == "two":
        trainer = TwoDataParallelTrainer(frame_mode)
        mpi_worker_multi_learner(trainer, args["outputdir"], "", args["mb_source"])

    elif args["trainer_type"] == "single":
        print("Coming to a single learner")
        trainer = SingleDataParallelTrainer(frame_mode)
        mpi_worker_multi_learner(trainer, args["outputdir"], "", args["mb_source"])

    cntk.distributed.Communicator.finalize()
