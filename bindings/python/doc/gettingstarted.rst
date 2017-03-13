Getting started 
===============
You can optionally try the `tutorials <https://notebooks.azure.com/cntk/libraries/tutorials>`__ with pre-installed CNTK running in Azure Notebook hosted environment (for free) if you have not installed the toolkit in your own machine.

If you have installed CNTK on your machine, after going through the `installation steps <https://github.com/Microsoft/CNTK/wiki/CNTK-Binary-Download-and-Configuration>`__, 
you can start using CNTK from Python right away (don't forget to ``activate`` your Python environment):

    >>> import cntk
    >>> cntk.__version__
    '2.0.beta12.0+'
    
    >>> cntk.minus([1, 2, 3], [4, 5, 6]).eval()
    array([-3., -3., -3.], dtype=float32)

The above makes use of the CNTK ``minus`` node with two array constants. Every operator has an ``eval()`` method that can be called which runs a forward 
pass for that node using its inputs, and returns the result of the forward pass. A slightly more interesting example that uses input variables (the 
more common case) is as follows:

    >>> import numpy as np
    >>> x = cntk.input_variable(2)
    >>> y = cntk.input_variable(2)
    >>> x0 = np.asarray([[2., 1.]], dtype=np.float32)
    >>> y0 = np.asarray([[4., 6.]], dtype=np.float32)
    >>> cntk.squared_error(x, y).eval({x:x0, y:y0})
    array([[ 29.]], dtype=float32)

In the above example we are first setting up two input variables with shape ``(1, 2)``. We then setup a ``squared_error`` node with those two variables as 
inputs. Within the ``eval()`` method we can setup the input-mapping of the data for those two variables. In this case we pass in two numpy arrays. 
The squared error is then of course ``(2-4)**2 + (1-6)**2 = 29``.

As the graph nodes implement the NumPy array interface, you can easily access
their content and use them in other NumPy operations:

    >>> import cntk as C
    >>> c = C.constant(3, shape=(2,3))
    >>> np.asarray(c)
    array([[ 3.,  3.,  3.],
           [ 3.,  3.,  3.]], dtype=float32)
    >>> np.ones_like(c)
    array([[ 1.,  1.,  1.],
           [ 1.,  1.,  1.]], dtype=float32)

Overview and first run
----------------------

CNTK2 is a major overhaul of CNTK in that one now has full control over the data and how it is read in, the training and testing loops, and minibatch 
construction. The Python bindings provide direct access to the created network graph, and data can be manipulated outside of the readers not only 
for more powerful and complex networks, but also for interactive Python sessions while a model is being created and debugged.

CNTK2 also includes a number of ready-to-extend examples and a layers library. The latter allows one to simply build a powerful deep network by 
snapping together building blocks such as convolution layers, recurrent neural net layers (LSTMs, etc.), and fully-connected layers. To begin, we will take a 
look at a standard fully connected deep network in our first basic use.

First basic use
~~~~~~~~~~~~~~~

The first step in training or running a network in CNTK is to decide which device it should be run on. If you have access to a GPU, training time 
can be vastly improved. To explicitly set the device to GPU, set the target device as follows::

    from cntk.device import set_default_device, gpu
    set_default_device(gpu(0))

Now let's setup a network that will learn a classifier based on the example fully connected classifier network 
(``nn.fully_connected_classifier_net``). This is defined, along with several other simple and more complex DNN building blocks in 
``Examples/common/nn.py``. Go to the ``[CNTK root]/Examples/common/`` directory and create a ``simplenet.py`` file with the 
following contents:

.. literalinclude:: simplenet.py

Running ``python simplenet.py`` (using the correct python environment) will generate this output::

      average      since    average      since      examples
         loss       last     metric       last
      ------------------------------------------------------
        0.693      0.693                                  25
        0.699      0.703                                  75
        0.727      0.747                                 175
        0.706      0.687                                 375
        0.687       0.67                                 775
        0.656      0.626                                1575
         0.59      0.525                                3175
        0.474      0.358                                6375
        0.359      0.245                               12775
         0.29      0.221                               25575
      error rate on an unseen minibatch: 0.0


The example above sets up a 2-layer fully connected deep neural network with 50 hidden dimensions per layer. We first setup two input variables, one for 
the input data and one for the labels. We then called the fully connected classifier network model function which simply sets up the required weights, 
biases, and activation functions for each layer.

We set two root nodes in the network: ``ce`` is the cross entropy which defined our model's loss function, and ``pe`` is the classification error. We 
set up a trainer object with the root nodes of the network and a learner. In this case we pass in the standard SGD learner with default parameters and a 
learning rate of 0.02.

Finally, we manually perform the training loop. We run through the data for the specific number of epochs (``num_minibatches_to_train``), get the ``features`` 
and ``labels`` that will be used during this training step, and call the trainer's ``train_minibatch`` function which maps the input and label variables that 
we setup previously to the current ``features`` and ``labels`` data (numpy arrays) that we are using in this minibatch. We use the convenience function 
``print_training_progress`` to display our loss and error every 20 steps and then finally we test our network again using the ``trainer`` object. It's 
as easy as that!

Now that we've seen some of the basics of setting up and training a network using the CNTK Python API, let's look at a more interesting deep 
learning problem in more detail (for the full example above along with the function to generate random data, please see 
``Tutorials/NumpyInterop/FeedForwardNet.py``).



