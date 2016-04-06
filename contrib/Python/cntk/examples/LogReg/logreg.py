import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from cntk import *

cur_dir = os.path.dirname(__file__)

# Using data from https://github.com/Microsoft/CNTK/wiki/Tutorial
train_file = os.path.join(cur_dir, "Train-3Classes.txt") 
test_file = os.path.join(cur_dir, "Test-3Classes.txt") 
mapping_file = os.path.join(cur_dir, "SimpleMapping-3Classes.txt")

def train_eval_logreg(criterion_name=None, eval_name=None):
    X = Input(2)
    y = Input(3)

    W = LearnableParameter(3, 2)
    b = LearnableParameter(3, 1)

    out = Times(W, X) + b
    out.tag = 'output'
    ce = CrossEntropyWithSoftmax(y, out, var_name=criterion_name)
    ce.tag = 'criterion'
    eval = SquareError(y, out, var_name=eval_name)
    eval.tag = 'eval'

    my_sgd = SGD(
        epoch_size=0, minibatch_size=25, learning_ratesPerMB=0.1, max_epochs=3)

    with Context('demo', root_nodes=[ce,eval], clean_up=False) as ctx:
        X.attach_uci_fast_reader(train_file, 0)
        y.attach_uci_fast_reader(train_file, 2, True, 1, mapping_file)
        ctx.train(my_sgd)
        
        X.attach_uci_fast_reader(test_file, 0)
        y.attach_uci_fast_reader(test_file, 2, True, 1, mapping_file)
        result = ctx.test()

        return result

def test_logreg():
    result = train_eval_logreg('crit_node', 'eval_node')
    assert result['SamplesSeen'] == 500
    assert result['Perplexity'] == 1.2216067
    assert result['eval_node'] == 13.779223
    assert result['crit_node'] == 0.20016696

if __name__ == "__main__":
    print(train_eval_logreg())
