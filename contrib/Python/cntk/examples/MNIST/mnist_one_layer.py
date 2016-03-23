import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from cntk import *

# =====================================================================================
# MNIST Example, one hidden layer neural network
# =====================================================================================

def dnn_sigmoid_layer(in_dim, out_dim, x, param_scale):
    W = LearnableParameter(out_dim, in_dim, initValueScale=param_scale) 
    b = LearnableParameter(out_dim, 1, initValueScale=param_scale) 
    t = Times(W, x)
    z = Plus(t, b)
    return Sigmoid(z)

def dnn_layer(in_dim, out_dim, x, param_scale):
    W = LearnableParameter(out_dim, in_dim, initValueScale=param_scale)
    b = LearnableParameter(out_dim, 1, initValueScale=param_scale)
    t = Times(W, x)
    return Plus(t, b)
    
if (__name__ == "__main__"):
    
    # Network definition 
    feat_dim=784
    label_dim=10
    hidden_dim=200
    
    training_filename=os.path.join("Data", "Train-28x28.txt")    
    
    features = Input(feat_dim, var_name='features')
    
    feat_scale = Constant(0.00390625)
    feats_scaled = Scale(feat_scale, features)
    
    labels = Input(label_dim, tag='label', var_name='labels')
    
    h1 = dnn_sigmoid_layer(feat_dim, hidden_dim, feats_scaled, 1)    
    out = dnn_layer(hidden_dim, label_dim, h1, 1)
    out.tag = 'output'
    
    ec = CrossEntropyWithSoftmax(labels, out)
    ec.tag = 'criterion'
    
    # Build the reader            
    r = UCIFastReader(filename=training_filename)
    
    # Add the input node to the reader
    r.add_input(features, 1, feat_dim)
    r.add_input(labels, 0, 1, label_dim, os.path.join("Data", "labelsmap.txt"))
    
    # Build the optimizer (settings are scaled down)
    my_sgd = SGD(epoch_size = 600, minibatch_size = 32, learning_ratesPerMB = 0.1, max_epochs = 5, momentum_per_mb = 0)
    
    # Create a context or re-use if already there
    with Context('mnist_one_layer', optimizer= my_sgd, root_node= ec, clean_up=False) as ctx:            
        # CNTK actions
        ctx.train(r)
        r["FileName"] = os.path.join("Data", "Test-28x28.txt")
        ctx.test(r)        
        ctx.predict(r)       
