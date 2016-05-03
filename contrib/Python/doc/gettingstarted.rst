Getting started
===============

Installation
------------
This page will guide you through the following three required steps:

#. Make sure that all Python requirements are met
#. Build and install CNTK
#. Install the Python API and set it up

Requirements
~~~~~~~~~~~~
You will need the following Python packages: 

:Python: 2.7+ or 3.3+
:NumPy: 1.10
:Scipy: 0.17

On Linux a simple ``pip install`` should suffice. On Windows, you will get
everything you need from `Anaconda <https://www.continuum.io/downloads>`_.

Installing CNTK
~~~~~~~~~~~~~~~
Please follow the instructions on `CNTK's GitHub page 
<https://github.com/Microsoft/CNTK/wiki/Setup-CNTK-on-your-machine>`_. 
After you have built the CNTK binary, find the build location. It will be 
something like ``<cntkpath>/x64/Debug_CpuOnly/cntk``. You will need this for 
the next step.

Installing the Python module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#. Go to ``<cntkpath>/contrib/Python`` and run ``python setup.py install``
#. Set up the environment variable ``CNTK_EXECUTABLE_PATH`` to point to the
   CNTK executable
#. Enjoy Python's ease of use with CNTK's speed::

>>> import cntk as cn
>>> cn.__version__
1.4
>>> with cn.Context('demo', clean_up=False) as ctx:
...     a = cn.constant([[1,2], [3,4]])
...     print(ctx.eval(a + [[10,20], [30, 40]]))
[[11.0, 22.0], [33.0, 44.0]]

In this case, we have set ``clean_up=False`` so that you can now peek into the
folder ``_cntk_demo`` and see what has been created under the hood for you.

Most likely, you will find issues or rough edges. Please help us improve CNTK
by posting any problems to https://github.com/Microsoft/CNTK/issues. Thanks!

Overview and first run
----------------------

CNTK is a powerful toolkit appropriate for everything from complex deep learning 
research to distributed production environment serving of learned models. It is 
also great for learning, however, and we will start with a basic regression example 
to get comfortable with the API. Then, we will look at an area where CNTK shines: 
working with sequences, where we will demonstrate state-of-the-art sequence classification 
with an LSTM (long short term memory network).

First basic use
~~~~~~~~~~~~~~~

The CNTK Python API allows users to easily define a computational network, define the data 
that will pass through the network, setup how learning should be performed, and finally, train 
and test the network. Here we will go through a simple example of using the CNTK Python API to 
learn to separate data into two classes. Following the code, some basic CNTK concepts will be 
explained::

    import cntk as C
    import numpy as np

    def simple_network():
        # 500 samples, 250-dimensional data
        N = 500
        d = 250

        # create synthetic data using numpy
        X = np.random.randn(N, d)
        Y = np.random.randint(size=(N, 1), low=0, high=2)
        Y = np.hstack((Y, 1-Y))

        # set up the training data for CNTK
        x = C.input_numpy(X, has_dynamic_axis=False)
        y = C.input_numpy(Y, has_dynamic_axis=False)

        # define our network parameters: a weight tensor and a bias
        W = C.ops.parameter((2, d))
        b = C.ops.parameter((2, 1))
		
        # create a dense 'layer' by multiplying the weight tensor and  
        # the features and adding the bias
        out = C.ops.times(W, x) + b

        # setup the criterion node using cross entropy with softmax
        ce = C.ops.cross_entropy_with_softmax(y, out)
        ce.tag = 'criterion'
        ce.name = 'loss'

        # define our SGD parameters and train!
        my_sgd = C.SGDParams(epoch_size=0, minibatch_size=25, learning_rates_per_mb=0.1, max_epochs=3)
        with C.LocalExecutionContext('logreg') as ctx:
            ctx.train(root_nodes=[ce], training_params=my_sgd)	        
            print(ctx.test(root_nodes=[ce]))


In the example above, we first create a synthetic data set of 500 samples, each with a 2-dimensional 
one-hot vector representing 0 (``[1 0]``) or 1 (``[0 1]``). We then begin describing the topology of our network 
by setting up the data inputs. This is typically done using the :class:`cntk.reader.CNTKTextFormatReader` by reading data 
in from a file, but for interactive experimentation and small examples we can use the ``input_numpy`` reader to 
access numpy data. Because dealing with dynamic axis data and sequences is where CNTK really shines, 
the default input data has a dynamic axis defined. Since we're not dealing with dynamic axes here, we 
set ``has_dynamic_axis`` to False.

Next, we define our network. In this case it's a simple 1-layer network with a weight tensor and a bias. 
We multiply our data `x` with the weight tensor `W` and add the bias `b`. We then input the model prediction 
into the `cross_entropy_with_softmax` node. This node first runs the data through a `softmax` to get 
probabilities for each class. Then the Cross Entropy loss function is applied. We tag the node `ce` with 
"criterion" so that CNTK knows it's a node from which the learning can start flowing back through the network.

Finally, we define our learning algorithm. In this case we use Stochastic Gradient Descent (SGD) and pass in 
some basic parameters. First, `epoch_size` allows different amounts of data per epoch. When we set it to 0, 
SGD looks at all of the training data in each epoch. Next, `minibatch_size` is the number of samples to look 
at for each minibatch; `learning_rates_per_mb` is the learning rate that SGD will use when the parameters are 
updated at the end of each minibatch; and `max_epochs` is the maximum number of epochs to train for.

The last step is to set up an execution context. An execution context can be either `Local` or `Deferred`. In the 
former case, as we use here, the methods (such as training and testing the network) are done locally and 
immediately so that the result is returned interactively to python. With a `Deferred` context, the methods simply 
set up a configuration file that can be used with CNTK at a later date. Here, with the local execution context, 
we train the network by passing in the root node and the optimizer we are using, and finally, we test its 
performance. Here is the output of the above example:

``{'SamplesSeen': 500, 'Perplexity': 1.1140191, 'loss': 0.10797427}``

Now that we've seen some of the basics of setting up and training a network using the CNTK Python API, 
let's look at a more interesting deep learning problem in more detail.


Sequence classification
~~~~~~~~~~~~~~~~~~~~~~~

One of the most exciting areas in deep learning is the powerful idea of recurrent 
neural networks (RNNs). RNNs are in some ways the Hidden Markov Models of the deep 
learning world. They are networks with loops in them and they allow us to model the 
current state given the result of a previous state. In other words, they allow information 
to persist.

A particular type of RNN -- the Long Short Term Memory (LSTM) network -- is exceedingly 
useful and in practice is what we commonly use when implementing an RNN. For more on why 
LSTMs are so powerful, see, e.g. http://colah.github.io/posts/2015-08-Understanding-LSTMs/. 
For our purposes, we will concentrate on the central feature of the LSTM model: the `memory 
cell`. 

.. image:: images/lstm_cell.png
    :width: 400px
    :alt: LSTM cell

The ...

In this example we can think of the LSTM as a layer being added to the network::

	def lstm_layer(output_dim, cell_dim, x, input_dim):    
    
		# use the CNTK operator `past_value` to get the previous state of the LSTM
		prev_state_h = past_value(0, 'lstm_state_h')
		prev_state_c = past_value(0, 'lstm_state_c')
        
		lstm_state_c, lstm_state_h = lstm_func(output_dim, cell_dim, x, input_dim, prev_state_h, prev_state_c)
		lstm_state_c.name = 'lstm_state_c'
		lstm_state_h.name = 'lstm_state_h'

		# return the hidden state
		return lstm_state_h


...

The parameters in an LSTM cell::

    def lstm_func(output_dim, cell_dim, x, input_dim, prev_state_h, prev_state_c):
        
        # input gate (t)
        it_w = times(parameter((cell_dim, input_dim)), x)
        it_b = parameter((cell_dim))
        it_h = times(parameter((cell_dim, output_dim)), prev_state_h)
        it_c = parameter((cell_dim)) * prev_state_c        
        it = sigmoid((it_w + it_b + it_h + it_c), name='it')

        # applied to tanh of input    
        bit_w = times(parameter((cell_dim, input_dim)), x)
        bit_h = times(parameter((cell_dim, output_dim)), prev_state_h)
        bit_b = parameter((cell_dim))
        bit = it * tanh(bit_w + (bit_h + bit_b))
        
        # forget-me-not gate (t)
        ft_w = times(parameter((cell_dim, input_dim)), x)
        ft_b = parameter((cell_dim))
        ft_h = times(parameter((cell_dim, output_dim)), prev_state_h)
        ft_c = parameter((cell_dim)) * prev_state_c        
        ft = sigmoid((ft_w + ft_b + ft_h + ft_c), name='ft')

        # applied to cell(t-1)
        bft = ft * prev_state_c
        
        # c(t) = sum of both
        ct = bft + bit
        
        # output gate
        ot_w = times(parameter((cell_dim, input_dim)), x)
        ot_b = parameter((cell_dim))
        ot_h = times(parameter((cell_dim, output_dim)), prev_state_h)
        ot_c = parameter((cell_dim)) * prev_state_c        
        ot = sigmoid((ot_w + ot_b + ot_h + ot_c), name='ot')
       
        # applied to tanh(cell(t))
        ht = ot * tanh(ct)
        
        # return cell value and hidden state
        return ct, ht

The above function ...
		

Operators
----------

Readers
----------
