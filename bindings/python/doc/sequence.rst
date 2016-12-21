Working with Sequences
=======

CNTK Concepts
~~~~~~~~~~~~~

CNTK inputs, outputs and parameters are organized as *tensors*. Each tensor has a *rank*:
A scalar is a tensor of rank 0, a vector is a tensor of rank 1, a matrix is a tensor 
of rank 2, and so on. We refer to these different dimensions as *axes*.

Every CNTK tensor has some *static axes* and some *dynamic axes*.
The static axes have the same length throughout the life of the network.
The dynamic axes are like static axes in that they define a meaningful grouping of the numbers contained in the tensor but:

 - their length can vary from instance to instance
 - their length is typically not known before each minibatch is presented
 - they may be ordered

A minibatch is also a tensor. Therefore, it has a dynamic axis, called the *batch axis*,
whose length can change from minibatch to minibatch. At the time of this writing 
CNTK supports a single additional dynamic axis. It is sometimes referred to as the sequence 
axis but it doesn't have a dedicated name. This axis enables working with
sequences in a high-level way. When operations on sequences are performed, CNTK
does a simple type-checking to determine if combining two sequences is always safe.

To make this more concrete, let's consider two examples. First, let's see
how a minibatch of short video clips is represented in CNTK. 
Suppose that the video clips are all 640x480 in 
resolution and they are shot in color which is typically encoded with three channels.
This means that our minibatch has three static axes of length 640, 480, and 3 respectively. 
It also has two dynamic axes:
the length of the video and the minibatch axis. So a minibatch of 16 videos each
of which is 240 frames long would be represented as a 16 x 240 x 3 x 640 x 480
tensor. 

Another example where dynamic axes provide an elegant solution is in learning to rank documents
given a query. Typically, the training data in this scenario consist of a set of 
queries, with each query having a variable number of associated documents. Each of the query-document
pairs includes a relevance judgment or label (e.g. whether the document is relevant for that query
or not). Now depending on how we treat the words in each document we can either place
them on a static axis or a dynamic axis. To place them on a static axis we can process
each document as a (sparse) vector of size equal to the size of our vocabulary
containing for each word (or short phrase) the number of times it appears in the
document. However we can also process the document to be a sequence of words
in which case we use another dynamic axis. In this case we have the following nesting:

 - Query: CNTK

   - Document 1:

     - Microsoft
     - Cognitive
     - Toolkit

   - Document 2:

     - Cartoon
     - Network

   - Document 3:

     - NVIDIA
     - Microsoft
     - Accelerate
     - AI

 - Query: flower

   - Document 1:

     - Flower
     - Wikipedia

   - Document 2:

     - Local 
     - Florist
     - Flower
     - Delivery

The outermost level is the batch axis. The document level should have 
a dynamic axis because we have a variable number of candidate documents per query. 
The innermost level should also have a dynamic axis because each document 
has a variable number of words. The tensor describing this minibatch will also
have one or more static axes, describing features such as the identity of the words in
the query and the document. With rich enough training data it is possible to have
another level of nesting, namely a session, in which multiple related queries belong
to.

Sequence classification
~~~~~~~~~~~~~~~~~~~~~~~

One of the most exciting areas in deep learning is the powerful idea of recurrent 
neural networks (RNNs). RNNs are in some ways the Hidden Markov Models of the deep 
learning world. They are networks that process variable length sequences using 
a fixed set of parameters. Therefore they have to learn to summarize all the
observations in the input sequence into a finite dimensional state, predict the
next observation using that state and transform their current state and the observed
input into their next state. In other words, they allow information 
to persist. So, while a traditional neural network layer can be thought of as having data 
flow through as in the figure on the left below, an RNN layer can be seen as the figure 
on the right.

.. figure:: images/nn_layers.png
    :width: 600px
    :alt: NN Layers

As is apparent from the figure above on the right, RNNs are the natural structure for 
dealing with sequences. This includes everything from text to music to video; anything 
where the current state is dependent on the previous state. While RNNs are indeed 
powerful, the "vanilla" RNN is extremely hard to learn via gradient based methods.
Because the gradient needs to flow back through the network to learn, the contribution 
from an early element (for example a word at the start of a sentence) on a much later 
elements (like the last word) can essentially vanish.

Dealing with the above problem is an active area of research. An architecture that
seems to be successful in practice is the Long Short Term Memory (LSTM) network. 
LSTMs are a type of RNN that are exceedingly useful and in practice are what we commonly 
use when implementing an RNN. A good explanation of the merits of LSTMs is at 
http://colah.github.io/posts/2015-08-Understanding-LSTMs. An LSTM is a 
a differentiable function that takes an input and a state and produces an output
and a new state.

In our example, we will be using an LSTM to do sequence classification. But for even 
better results, we will also introduce an additional concept here: 
`word embeddings <https://en.wikipedia.org/wiki/Word_embedding>`_. 
In traditional NLP approaches, words are identified with the standard basis of a high 
dimensional space (the vocabulary) (the first word is (1, 0, 0, ...), the second one 
is (0, 1, 0, ...) and so on).  Each word is orthogonal to all others. But that is not 
a good abstraction.  In real languages, some words are very similar (we call them 
synonyms) or they function in similar ways (e.g. Paris, Seattle, Tokyo). The key 
observation is that words that appear in similar contexts should be similar. We
can let a neural network sort out these details by forcing each word to be represented 
by a short learned vector.  Then in order for the network to do well on its task it 
has to learn to map the words to these vectors effectively. For example, the vector 
representing the word "cat" may somehow be close, in some sense, to the vector for "dog". 
In our task, we will use a pre-computed word embedding model 
using `GloVe <http://nlp.stanford.edu/projects/glove/>`_ and each of the words in the
sequences will be replaced by their respective GloVe vector.

Now that we've decided on our word representation and the type of recurrent neural 
network we want to use, let's define the network that we'll use to do 
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

That is the entire network definition. We now simply set up our criterion nodes and then our training loop. In the above example we use a minibatch
size of 200 and use basic SGD with the default parameters and a small learning rate of 0.0005. This results in a powerful state-of-the-art model for 
sequence classification that can scale with huge amounts of training data. Note that as your training data size grows, you should give more capacity to 
your LSTM by increasing the number of hidden dimensions. Further, you can get an even more complex network by stacking layers of LSTMs. This is also easy 
using the LSTM layer function [coming soon].
