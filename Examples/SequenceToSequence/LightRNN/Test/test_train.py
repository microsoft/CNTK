import os
import sys
import subprocess

abs_path = os.path.dirname(os.path.abspath(__file__))
work_dir = os.path.join(abs_path, '..', 'LightRNN')
sys.path.append(work_dir)
os.chdir(work_dir)


def run_command(**kwargs):
    command = ['python', os.path.join(abs_path, '..', 'LightRNN', 'train.py')]
    for key, value in kwargs.items():
        command += ['-' + key, str(value)]
    subprocess.call(command)


def run_distributed_command(**kwargs):
    command = ['mpiexec', '-n', '2', 'python', os.path.join(abs_path, '..', 'LightRNN', 'train_distributed.py')]
    for key, value in kwargs.items():
        command += ['-' + key, str(value)]
    subprocess.call(command)


def test_train():
    run_command(datadir=os.path.join(abs_path, '..', 'PTB', 'Data'),
                outputdir=os.path.join(abs_path, '..', 'LightRNN'),
                vocabdir=os.path.join(abs_path, '..', 'PTB', 'Allocation'),
                vocab_file=os.path.join(abs_path, '..', 'PTB', 'Allocation', 'vocab.txt'),
                alloc_file=os.path.join(abs_path, '..', 'PTB', 'Allocation', 'word-0.location'),
                vocabsize=10000,
                optim='adam', lr=0.1,
                embed=500, nhid=500, batchsize=20, layer=2,
                epochs=1)
    """ result
        Training 4204202 parameters in 14 parameter tensors.
        Epoch  1: Minibatch [    1 -   100 ], loss = 8.699124, error = 1.827000, speed = 7111 tokens/s
        Epoch  1: Minibatch [  101 -   200 ], loss = 8.361985, error = 1.765305, speed = 7529 tokens/s
        Epoch  1: Minibatch [  201 -   300 ], loss = 8.104901, error = 1.714703, speed = 7680 tokens/s
        Epoch  1: Minibatch [  301 -   400 ], loss = 7.916897, error = 1.676723, speed = 7529 tokens/s
        Epoch  1: Minibatch [  401 -   500 ], loss = 7.770832, error = 1.645750, speed = 7619 tokens/s
        Epoch  1: Minibatch [  501 -   600 ], loss = 7.645920, error = 1.618013, speed = 7680 tokens/s
        Epoch  1: Minibatch [  601 -   700 ], loss = 7.550490, error = 1.596683, speed = 7593 tokens/s
        Epoch  1: Minibatch [  701 -   800 ], loss = 7.460805, error = 1.577678, speed = 7641 tokens/s
        Epoch  1: Minibatch [  801 -   900 ], loss = 7.384566, error = 1.561389, speed = 7680 tokens/s
        Epoch  1: Minibatch [  901 -  1000 ], loss = 7.323613, error = 1.547836, speed = 7619 tokens/s
        Epoch  1: Minibatch [ 1001 -  1100 ], loss = 7.271085, error = 1.536565, speed = 7652 tokens/s
        Epoch  1: Minibatch [ 1101 -  1200 ], loss = 7.220058, error = 1.525594, speed = 7680 tokens/s
        Epoch  1: Minibatch [ 1201 -  1300 ], loss = 7.176397, error = 1.516563, speed = 7633 tokens/s
        Epoch  1 Done : Valid error = 6.548713, Test error = 6.534246
    """

    run_distributed_command(
                datadir=os.path.join(abs_path, '..', 'PTB', 'Data'),
                outputdir=os.path.join(abs_path, '..', 'LightRNN'),
                vocabdir=os.path.join(abs_path, '..', 'PTB', 'Allocation'),
                vocab_file=os.path.join(abs_path, '..', 'PTB', 'Allocation', 'vocab.txt'),
                vocabsize=10000,
                optim='adam', lr=0.5,
                embed=1500, nhid=1500, batchsize=20, layer=2,
                epochs=1)
    """ result
        Epoch  1: Minibatch [    1 -   100 ], loss = 8.252628, error = 1.709711, speed = 2415 tokens/s
        Epoch  1: Minibatch [  101 -   200 ], loss = 7.699344, error = 1.615191, speed = 2438 tokens/s
        Epoch  1: Minibatch [  201 -   300 ], loss = 7.383088, error = 1.553174, speed = 2445 tokens/s
        Epoch  1: Minibatch [  301 -   400 ], loss = 7.170318, error = 1.513025, speed = 2449 tokens/s
        Epoch  1: Minibatch [  401 -   500 ], loss = 7.012431, error = 1.483406, speed = 2461 tokens/s
        Epoch  1: Minibatch [  501 -   600 ], loss = 6.891491, error = 1.461219, speed = 2461 tokens/s
        Epoch  1 Done : Valid error = 6.131400, Test error = 6.111777
    """


if __name__ == '__main__':
    test_train()
