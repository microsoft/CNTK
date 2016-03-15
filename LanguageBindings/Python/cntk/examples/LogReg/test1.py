import sys
sys.path.insert(0, r"F:\cntk\LanguageBindings\Python")

from cntk import *

if (__name__ == "__main__"):
    x = Input(2)
    y = Input(3, tag='label')
    w = LearnableParameter(3, 2)
    b = LearnableParameter(3, 1)
    t = Times(w, x)
    out = Plus(t, b)
    out.tag = 'output'
    ec = CrossEntropyWithSoftmax(y, out)
    ec.tag = 'criterion'

    reader = UCIFastReader(
        "Train-3Classes.txt", "2", "0", "1", "2", "3", "SimpleMapping-3Classes.txt")
    my_sgd = SGD(
        epoch_size=0, minibatch_size=25, learning_ratesPerMB=0.1, max_epochs=3)
    with Context('demo', optimizer=my_sgd, root_node=ec, clean_up=False) as ctx:
        ctx.train({ec: (reader, (0, 2))})
