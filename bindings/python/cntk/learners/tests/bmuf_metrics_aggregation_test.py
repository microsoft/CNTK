"""
This tests the metrics averaging functionality in BMUF. All workers should be reporting the same loss and eval metrics.
"""
import pytest
import cntk
import numpy as np
import sys, os
sys.path.append(os.path.dirname(__file__))
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

class SimpleBMUFTrainer():
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
            learner = cntk.block_momentum_distributed_learner(cntk.momentum_sgd(self.output.parameters, cntk.learning_parameter_schedule(0.0001), cntk.momentum_as_time_constant_schedule(1000)), 
                                                              block_size=1000, block_learning_rate=0.01, block_momentum_as_time_constant=1000)

            comm_rank = cntk.distributed.Communicator.rank()
            self.trainer = cntk.Trainer(self.output, (self.ce, self.err), [learner], [cntk.logging.ProgressPrinter(freq=progress_freq, tag="Training", rank=comm_rank)])
        except RuntimeError:
            self.trainer = None
        return

def get_minibatch(bmuf, working_dir, mb_source, num_data_partitions=1, partition_index=0):
    from cntk.io import MinibatchSource, CTFDeserializer, StreamDef, StreamDefs

    if mb_source == "numpy":
        assert(num_data_partitions == 1) # numpy option does not support more than one partition in this impl.
        assert(partition_index == 0) 
        for i in range(num_batches):
            features = []
            labels = []
            for j in range(batch_size):
                seq_len_j = [seq_len, seq_len + 5, seq_len - 5][j % 3]
                x = np.random.rand( seq_len_j, feat_dim).astype(np.float32)
                y = np.random.rand( seq_len_j, label_dim).astype(np.float32)
                features.append(x)    
                labels.append(y)
            yield {bmuf.feat: features, bmuf.label: labels}

    if mb_source in ("ctf_utterance", "ctf_frame", "ctf_bptt"):
        if mb_source == "ctf_frame":
            #frame mode data without sequence ids.
            ctf_data = ctf_data = '''\
|S0 0.49  0.18  0.84  0.7   0.59 |S1 0.12  0.24  0.14
|S0 0.69  0.63  0.47  0.93  0.69 |S1 0.34  0.85  0.17
|S0 0.04  0.5   0.39  0.86  0.28 |S1 0.62  0.36  0.53
|S0 0.71  0.9   0.15  0.83  0.18 |S1 0.2   0.74  0.04
|S0 0.38  0.67  0.46  0.53  0.75 |S1 0.6   0.14  0.35
|S0 0.94  0.54  0.09  0.55  0.08 |S1 0.07  0.53  0.47
|S0 0.11  0.24  0.17  0.72  0.72 |S1 0.9   0.98  0.18
|S0 0.3   1.    0.34  0.06  0.78 |S1 0.15  0.69  0.63
|S0 0.69  0.86  0.59  0.49  0.99 |S1 0.13  0.6   0.21
'''
        #sequence mode data with sequence id
        else:
            ctf_data = ctf_data = '''\
0	|S0 0.49  0.18  0.84  0.7   0.59 |S1 0.12  0.24  0.14
0	|S0 0.69  0.63  0.47  0.93  0.69 |S1 0.34  0.85  0.17
0	|S0 0.04  0.5   0.39  0.86  0.28 |S1 0.62  0.36  0.53
0	|S0 0.71  0.9   0.15  0.83  0.18 |S1 0.2   0.74  0.04
0	|S0 0.38  0.67  0.46  0.53  0.75 |S1 0.6   0.14  0.35
0	|S0 0.94  0.54  0.09  0.55  0.08 |S1 0.07  0.53  0.47
0	|S0 0.11  0.24  0.17  0.72  0.72 |S1 0.9   0.98  0.18
2	|S0 0.3   1.    0.34  0.06  0.78 |S1 0.15  0.69  0.63
2	|S0 0.69  0.86  0.59  0.49  0.99 |S1 0.13  0.6   0.21
'''

        ctf_file = os.path.join(working_dir, '2seqtest.txt')
        with open(ctf_file, 'w') as f:
            f.write(ctf_data)

        # ctf_utterance model
        frame_mode = False
        truncation_length = 0

        if mb_source == "ctf_frame":
            frame_mode = True
        elif mb_source == "ctf_bptt":
            truncation_length = 2

        mbs = MinibatchSource(CTFDeserializer(ctf_file, StreamDefs(
            features  = StreamDef(field='S0', shape=feat_dim,  is_sparse=False),
            labels    = StreamDef(field='S1', shape=label_dim,  is_sparse=False)
        )), randomize=False, max_samples = batch_size*num_batches, 
            frame_mode=frame_mode, truncation_length=truncation_length)

        for i in range(num_batches):
            minibatch = mbs.next_minibatch(
                minibatch_size_in_samples=batch_size, 
                input_map={bmuf.feat: mbs.streams.features, bmuf.label: mbs.streams.labels}, 
                num_data_partitions=num_data_partitions, 
                partition_index=partition_index
            )
            if not minibatch:
                break
            yield minibatch

def mpi_worker(working_dir, mb_source, gpu):
    comm_rank = cntk.distributed.Communicator.rank()
    np.random.seed(comm_rank)

    if gpu:
        # test with only one GPU
        cntk.try_set_default_device(cntk.gpu(0))

    frame_mode = (mb_source == "ctf_frame")
    bmuf = SimpleBMUFTrainer(frame_mode)
    for i, data in enumerate(get_minibatch(bmuf, working_dir, mb_source)):        
        bmuf.trainer.train_minibatch(data)        
        if i % 50 == 0:
            bmuf.trainer.summarize_training_progress()
    print("SAMPLES %d"%(bmuf.trainer.total_number_of_samples_seen))

MB_SOURCES = ["numpy", "ctf_utterance", "ctf_frame", "ctf_bptt"]
#MB_SOURCES = ["numpy"]    
@pytest.mark.parametrize("mb_source", MB_SOURCES)
def test_bmuf_correct_metrics_averaging(tmpdir, device_id, mb_source):
    if platform.system() == 'Linux':
        pytest.skip('test only runs on Windows due to mpiexec -l option')

    # check whether trainer can be initialized or not
    bmuf = SimpleBMUFTrainer()
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-outputdir', '--outputdir')
    parser.add_argument('-mb_source', '--mb_source')
    parser.add_argument('-gpu', '--gpu', action='store_true')
    args = vars(parser.parse_args())

    mpi_worker(args["outputdir"], args["mb_source"], args["gpu"])    
    cntk.distributed.Communicator.finalize()
