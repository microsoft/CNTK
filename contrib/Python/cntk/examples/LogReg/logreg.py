import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from cntk import *

if (__name__ == "__main__"):
    X = Input(2)
    X.attach_uci_fast_reader("Train-3Classes.txt", 0)

    y = Input(3)
    y.attach_uci_fast_reader(
        "Train-3Classes.txt", 2, True, 1, "SimpleMapping-3Classes.txt")

    W = LearnableParameter(3, 2)
    b = LearnableParameter(3, 1)

    out = Times(W, X) + b
    out.tag = 'output'
    ce = CrossEntropyWithSoftmax(y, out)
    ce.tag = 'criterion'

    my_sgd = SGD(
        epoch_size=0, minibatch_size=25, learning_ratesPerMB=0.1, max_epochs=3)

    with Context('demo', root_node=ce, clean_up=False) as ctx:
        ctx.train(my_sgd, None)

        result = ctx.eval(out)
        print(result.argmax(axis=1))
