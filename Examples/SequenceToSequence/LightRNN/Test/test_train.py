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
     Epoch  1: Minibatch [    1 -   100], loss = 8.648669, error = 1.824985, speed = 5386 tokens/s 
     Epoch  1: Minibatch [  101 -   200], loss = 8.264636, error = 1.749534, speed = 5360 tokens/s
     Epoch  1: Minibatch [  201 -   300], loss = 8.006456, error = 1.696859, speed = 5351 tokens/s
     Epoch  1: Minibatch [  301 -   400], loss = 7.822579, error = 1.657684, speed = 5346 tokens/s
     Epoch  1: Minibatch [  401 -   500], loss = 7.683199, error = 1.625727, speed = 5344 tokens/s
     Epoch  1: Minibatch [  501 -   600], loss = 7.561859, error = 1.598211, speed = 5342 tokens/s
     Epoch  1: Minibatch [  601 -   700], loss = 7.469755, error = 1.577648, speed = 5340 tokens/s
     Epoch  1: Minibatch [  701 -   800], loss = 7.380890, error = 1.559256, speed = 5340 tokens/s
     Epoch  1: Minibatch [  801 -   900], loss = 7.308160, error = 1.544031, speed = 5339 tokens/s
     Epoch  1: Minibatch [  901 -  1000], loss = 7.250702, error = 1.531695, speed = 5338 tokens/s
     Epoch  1: Minibatch [ 1001 -  1100], loss = 7.199289, error = 1.520795, speed = 5338 tokens/s
     Epoch  1: Minibatch [ 1101 -  1200], loss = 7.151837, error = 1.510724, speed = 5337 tokens/s
     Epoch  1: Minibatch [ 1201 -  1300], loss = 7.110386, error = 1.502245, speed = 5337 tokens/s
     Epoch  1 Done : Valid error = 6.510330, Test error = 6.494924
    """

    run_command(datadir=os.path.join(abs_path, '..', 'PTB', 'Data'),
                outputdir=os.path.join(abs_path, '..', 'LightRNN'),
                vocabdir=os.path.join(abs_path, '..', 'PTB', 'Allocation'),
                vocab_file=os.path.join(abs_path, '..', 'PTB', 'Allocation', 'vocab.txt'),
                vocabsize=10000,
                optim='adam', lr=0.5,
                embed=1500, nhid=1500, batchsize=20, layer=2,
                epochs=1)
    """ result
    Epoch  1: Minibatch [    1 -   100 ], loss = 7.924267, error = 1.649892, speed = 1576 tokens/s 
    Epoch  1: Minibatch [  101 -   200 ], loss = 7.384230, error = 1.549129, speed = 1588 tokens/s
    Epoch  1: Minibatch [  201 -   300 ], loss = 7.130919, error = 1.503224, speed = 1579 tokens/s
    Epoch  1: Minibatch [  301 -   400 ], loss = 6.971337, error = 1.474891, speed = 1584 tokens/s
    Epoch  1: Minibatch [  401 -   500 ], loss = 6.844049, error = 1.452492, speed = 1587 tokens/s
    Epoch  1: Minibatch [  501 -   600 ], loss = 6.737159, error = 1.432615, speed = 1582 tokens/s
    Epoch  1: Minibatch [  601 -   700 ], loss = 6.647292, error = 1.416474, speed = 1585 tokens/s
    Epoch  1: Minibatch [  701 -   800 ], loss = 6.560721, error = 1.400702, speed = 1587 tokens/s
    Epoch  1: Minibatch [  801 -   900 ], loss = 6.487198, error = 1.387481, speed = 1584 tokens/s
    Epoch  1: Minibatch [  901 -  1000 ], loss = 6.426224, error = 1.376394, speed = 1585 tokens/s
    Epoch  1: Minibatch [ 1001 -  1100 ], loss = 6.374137, error = 1.366813, speed = 1587 tokens/s
    Epoch  1: Minibatch [ 1101 -  1200 ], loss = 6.321806, error = 1.356923, speed = 1584 tokens/s
    Epoch  1: Minibatch [ 1201 -  1300 ], loss = 6.272898, error = 1.348028, speed = 1585 tokens/s
    Epoch  1 Done : Valid error = 5.625651, Test error = 5.599864
    """


if __name__ == '__main__':
    test_train()
