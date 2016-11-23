Getting started 
===============

After going through the `installation steps <https://github.com/Microsoft/CNTK/wiki/CNTK-Binary-Download-and-Configuration>`__, 
you can start using CNTK from Python right away (don't forget to ``activate`` your Python environment):

    >>> import cntk
    >>> cntk.__version__
    '2.0'
    
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
following contents::

	import numpy as np
	import cntk
	import cntk.ops as C
	from cntk.learner import sgd, learning_rate_schedule, UnitType
	from cntk.utils import get_train_loss
	from nn import fully_connected_classifier_net
	from cntk.utils import ProgressPrinter

	def generate_random_data(sample_size, feature_dim, num_classes):
		 # Create synthetic data using NumPy.
		 Y = np.random.randint(size=(sample_size, 1), low=0, high=num_classes)

		 # Make sure that the data is separable
		 X = (np.random.randn(sample_size, feature_dim) + 3) * (Y + 1)
		 X = X.astype(np.float32)
		 # converting class 0 into the vector "1 0 0",
		 # class 1 into vector "0 1 0", ...
		 class_ind = [Y == class_number for class_number in range(num_classes)]
		 Y = np.asarray(np.hstack(class_ind), dtype=np.float32)
		 return X, Y

	def ffnet():
		inputs = 2
		outputs = 2
		layers = 2
		hidden_dimension = 50

		# Input variables denoting the features and label data
		input = C.input_variable((inputs), np.float32)
		label = C.input_variable((outputs), np.float32)

		# Instantiate the feedforward classification model
		z = fully_connected_classifier_net(input, outputs, hidden_dimension, layers, C.sigmoid)

		ce = C.cross_entropy_with_softmax(z, label)
		pe = C.classification_error(z, label)

		# Instantiate the trainer object to drive the model training
		lr_per_minibatch = learning_rate_schedule(0.125, UnitType.minibatch)
		trainer = cntk.Trainer(z, ce, pe, [sgd(z.parameters, lr=lr_per_minibatch)])

		# Get minibatches of training data and perform model training
		minibatch_size = 25
		num_minibatches_to_train = 1024

		pp = ProgressPrinter(0)
		for i in range(num_minibatches_to_train):
			features, labels = generate_random_data(minibatch_size, inputs, outputs)
			# Specify the mapping of input variables in the model to actual minibatch data to be trained with
			trainer.train_minibatch({input : features, label : labels})
			pp.update_with_trainer(trainer)
		test_features, test_labels = generate_random_data(minibatch_size, inputs, outputs)
		avg_error = trainer.test_minibatch({input : test_features, label : test_labels})
		print(' error rate on an unseen minibatch: {}'.format(avg_error))

	np.random.seed(98052)
	ffnet()

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


Sequence classification
~~~~~~~~~~~~~~~~~~~~~~~

One of the most exciting areas in deep learning is the powerful idea of recurrent 
neural networks (RNNs). RNNs are in some ways the Hidden Markov Models of the deep 
learning world. They are networks with loops in them and they allow us to model the 
current state given the result of a previous state. In other words, they allow information 
to persist. So, while a traditional neural network layer can be thought of as having data 
flow through as in the figure on the left below, an RNN layer can be seen as the figure 
on the right.

.. figure:: images/nn_layers.png
    :width: 600px
    :alt: NN Layers

As is apparent from the figure above on the right, RNNs are the natural structure for 
dealing with sequences. This includes everything from text to music to video; anything 
where the current state is dependent on the previous state. While RNNs are indeed 
powerful, the "vanilla" RNN suffers from an important problem: long-term dependencies. 
Because the gradient needs to flow back through the network to learn, the contribution 
from an early element (for example a word at the start of a sentence) on a much later 
elements (like the last word) can essentially vanish.

To deal with the above problem, we turn to the Long Short Term Memory (LSTM) network. 
LSTMs are a type of RNN that are exceedingly useful and in practice are what we commonly 
use when implementing an RNN. For more on why LSTMs are so powerful, see, e.g. 
http://colah.github.io/posts/2015-08-Understanding-LSTMs. For our purposes, we will 
concentrate on the central feature of the LSTM model: the `memory cell`. 

.. figure:: images/lstm_cell.png
    :width: 400px
    :alt: LSTM cell

    An LSTM cell.

The LSTM cell is associated with three gates that control how information is stored / 
remembered in the LSTM. The "forget gate" determines what information should be kept 
after a single element has flowed through the network. It makes this determination 
using data for the current time step and the previous hidden state. 

The "input gate" uses the same information as the forget gate, but passes it through 
a `tanh` to determine what to add to the state. The final gate is the "output gate" 
and it modulates what information should be output from the LSTM cell. This time we 
also take the previous state's value into account in addition to the previous hidden 
state and the data of the current state. We have purposely left the full details out 
for conciseness, so please see the link above for a full understanding of how an LSTM 
works.

In our example, we will be using an LSTM to do sequence classification. But for even 
better results, we will also introduce an additional concept here: 
`word embeddings <https://en.wikipedia.org/wiki/Word_embedding>`_. 
In traditional NLP approaches, words are seen as single points in a high dimensional 
space (the vocabulary). A word is represented by an arbitrary id and that single number 
contains no information about the meaning of the word or how it is used. However, with 
word embeddings each word is represented by a learned vector that has some meaning. For 
example, the vector representing the word "cat" may somehow be close, in some sense, to 
the vector for "dog", and each dimension is encoding some similarities or differences 
between those words that were learned usually by analyzing a large corpus. In our task, 
we will use a pre-computed word embedding model (e.g. from `GloVe <http://nlp.stanford.edu/projects/glove/>`_) 
and each of the words in the sequences will be replaced by their respective GloVe vector.

Now that we've decided on our word representation and the type of recurrent neural 
network we want to use, let's define the computational network that we'll use to do 
sequence classification. We can think of the network as adding a series of layers:

1. Embedding layer (individual words in each sequence become vectors)
2. LSTM layer (allow each word to depend on previous words)
3. Softmax layer (an additional set of parameters and output probabilities per class)

This network is defined as part of the example at ``Examples/SequenceClassification/SimpleExample/Python/SequenceClassification.py``. Let's go through some 
key parts of the code::

    # model
    input_dim = 2000
    cell_dim = 25
    hidden_dim = 25
    embedding_dim = 50
    num_output_classes = 5

    # Input variables denoting the features and label data
    features = input_variable(shape=input_dim, is_sparse=True)
    label = input_variable(num_output_classes, dynamic_axes = [Axis.default_batch_axis()])

    # Instantiate the sequence classification model
    classifier_output = LSTM_sequence_classifer_net(features, num_output_classes, embedding_dim, hidden_dim, cell_dim)

    ce = cross_entropy_with_softmax(classifier_output, label)
    pe = classification_error(classifier_output, label)

    rel_path = r"../../../../Tests/EndToEndTests/Text/SequenceClassification/Data/Train.ctf"
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), rel_path)

    mb_source = text_format_minibatch_source(path, [
                    StreamConfiguration( 'features', input_dim, True, 'x' ),
                    StreamConfiguration( 'labels', num_output_classes, False, 'y')], 0)

    features_si = mb_source.stream_info(features)
    labels_si = mb_source.stream_info(label)

    # Instantiate the trainer object to drive the model training
    trainer = Trainer(classifier_output, ce, pe, [sgd_learner(classifier_output.parameters(), lr=0.0005)])

    # Get minibatches of sequences to train with and perform model training
    minibatch_size = 200
    training_progress_output_freq = 10
    i = 0
    while True:
        mb = mb_source.get_next_minibatch(minibatch_size)
        if  len(mb) == 0:
            break

        # Specify the mapping of input variables in the model to actual minibatch data to be trained with
        arguments = {features : mb[features_si].m_data, label : mb[labels_si].m_data}
        trainer.train_minibatch(arguments)

        print_training_progress(trainer, i, training_progress_output_freq)
        i += 1

Let's go through some of the intricacies of the network definition above. As usual, we first set the parameters of our model. In this case we 
have a vocab (input dimension) of 2000, LSTM hidden and cell dimensions of 25, an embedding layer with dimension 50, and we have 5 possible 
classes for our sequences. As before, we define two input variables: one for the features, and for the labels. We then instantiate our model. The 
``LSTM_sequence_classifier_net`` is a simple function which looks up our input in an embedding matrix and returns the embedded representation, puts 
that input through an LSTM recurrent neural network layer, and returns a fixed-size output from the LSTM by selecting the last hidden state of the 
LSTM::

    embedding_function = embedding(input, embedding_dim)
    LSTM_function = LSTMP_component_with_self_stabilization(embedding_function.output(), LSTM_dim, cell_dim)[0]
    thought_vector = select_last(LSTM_function)

    return linear_layer(thought_vector, num_output_classes)

That is the entire network definition. We now simply setup our criterion nodes and then setup our training loop. In the above example we use a minibatch 
size of 200 and use basic SGD with the default parameters and a small learning rate of 0.0005. This results in a powerful state-of-the-art model for 
sequence classification that can scale with huge amounts of training data. Note that as your training data size grows, you should give more capacity to 
your LSTM by increasing the number of hidden dimensions. Further, you can get an even more complex network by stacking layers of LSTMs. This is also easy 
using the LSTM layer function [coming soon].
