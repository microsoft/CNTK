
Hands-On Lab: Language Understanding with Recurrent Networks
============================================================

This hands-on lab shows how to implement a recurrent network to process
text, for the `Air Travel Information
Services <https://catalog.ldc.upenn.edu/LDC95S26>`__ (ATIS) task of slot
tagging (tag individual words to their respective classes, where the
classes are provided as labels in the the training data set). We will
start with a straight-forward embedding of the words followed by a
recurrent LSTM. We will then extend it to include neighbor words and run
bidirectionally. Lastly, we will turn this system into an intent
classifier.

The techniques you will practice are:

-  model description by composing layer blocks, a convenient way to
   compose networks/models without requiring the need to write formulas,
-  creating your own layer block
-  variables with different sequence lengths in the same network
-  training the network

We assume that you are familiar with basics of deep learning, and these
specific concepts:

-  recurrent networks (`Wikipedia
   page <https://en.wikipedia.org/wiki/Recurrent_neural_network>`__)
-  text embedding (`Wikipedia
   page <https://en.wikipedia.org/wiki/Word_embedding>`__)

Prerequisites
~~~~~~~~~~~~~

We assume that you have already `installed
CNTK <https://www.cntk.ai/pythondocs/setup.html>`__. This tutorial
requires CNTK V2. We strongly recommend to run this tutorial on a
machine with a capable CUDA-compatible GPU. Deep learning without GPUs
is not fun.

Downloading the data
^^^^^^^^^^^^^^^^^^^^

In this tutorial we are going to use a (lightly preprocessed) version of
the ATIS dataset. You can download the data automatically by running the
cell below or (in case it fails) by executing the manual instructions.

Fallback manual instructions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Please download the ATIS
`training <https://github.com/Microsoft/CNTK/blob/master/Tutorials/SLUHandsOn/atis.train.ctf>`__
and
`test <https://github.com/Microsoft/CNTK/blob/master/Tutorials/SLUHandsOn/atis.test.ctf>`__
files and put them at the same folder as this notebook. If you want to
see how the model is predicting on new sentences you type will also need
the vocabulary files for
`queries <https://github.com/Microsoft/CNTK/blob/master/Examples/Text/ATIS/query.wl>`__
and
`slots <https://github.com/Microsoft/CNTK/blob/master/Examples/Text/ATIS/slots.wl>`__

.. code:: python

    import requests
    
    def download(url, filename):
        """ utility to download necessary data """
        response = requests.get(url, stream=True)
        with open(filename, "wb") as handle:
            for data in response.iter_content():
                handle.write(data)
    
    url1 = "https://github.com/Microsoft/CNTK/blob/master/Examples/Tutorials/SLUHandsOn/atis.%s.ctf?raw=true"
    url2 = "https://github.com/Microsoft/CNTK/blob/master/Examples/Text/ATIS/%s.wl?raw=true"
    urls = [url1%"train", url1%"test", url2%"query", url2%"slots"]
    
    for t in urls:
        filename = t.split('/')[-1].split('?')[0]
        try:
            f = open(filename)
            f.close()
        except IOError:
            download(t, filename)
    

Importing CNTK and other useful libraries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CNTK is a python module that contains several submodules like ``io``,
``learner``, and ``layers``. We also use NumPy in some cases since the
results returned by CNTK work like NumPy arrays

.. code:: python

    import math
    import numpy as np
    from cntk.blocks import default_options, LSTM, Placeholder, Input        # building blocks
    from cntk.layers import Embedding, Recurrence, Dense, BatchNormalization # layers
    from cntk.models import Sequential                                       # higher level things
    from cntk.utils import ProgressPrinter, log_number_of_parameters
    from cntk.io import MinibatchSource, CTFDeserializer
    from cntk.io import StreamDef, StreamDefs, INFINITELY_REPEAT, FULL_DATA_SWEEP
    from cntk import future_value, combine, Trainer, cross_entropy_with_softmax, classification_error, splice
    from cntk.learner import adam_sgd, learning_rate_schedule

Task and Model Structure
------------------------

The task we want to approach in this tutorial is slot tagging. We use
the `ATIS corpus <https://catalog.ldc.upenn.edu/LDC95S26>`__. ATIS
contains human-computer queries from the domain of Air Travel
Information Services, and our task will be to annotate (tag) each word
of a query whether it belongs to a specific item of information (slot),
and which one.

The data in your working folder has already been converted into the
"CNTK Text Format." Let's look at an example from the test-set file
``atis.test.ctf``:

::

    19  |S0 178:1 |# BOS      |S1 14:1 |# flight  |S2 128:1 |# O
    19  |S0 770:1 |# show                         |S2 128:1 |# O
    19  |S0 429:1 |# flights                      |S2 128:1 |# O
    19  |S0 444:1 |# from                         |S2 128:1 |# O
    19  |S0 272:1 |# burbank                      |S2 48:1  |# B-fromloc.city_name
    19  |S0 851:1 |# to                           |S2 128:1 |# O
    19  |S0 789:1 |# st.                          |S2 78:1  |# B-toloc.city_name
    19  |S0 564:1 |# louis                        |S2 125:1 |# I-toloc.city_name
    19  |S0 654:1 |# on                           |S2 128:1 |# O
    19  |S0 601:1 |# monday                       |S2 26:1  |# B-depart_date.day_name
    19  |S0 179:1 |# EOS                          |S2 128:1 |# O

This file has 7 columns:

-  a sequence id (19). There are 11 entries with this sequence id. This
   means that sequence 19 consists of 11 tokens;
-  column ``S0``, which contains numeric word indices;
-  a comment column denoted by ``#``, to allow a human reader to know
   what the numeric word index stands for; Comment columns are ignored
   by the system. ``BOS`` and ``EOS`` are special words to denote
   beginning and end of sentence, respectively;
-  column ``S1`` is an intent label, which we will only use in the last
   part of the tutorial;
-  another comment column that shows the human-readable label of the
   numeric intent index;
-  column ``S2`` is the slot label, represented as a numeric index; and
-  another comment column that shows the human-readable label of the
   numeric label index.

The task of the neural network is to look at the query (column ``S0``)
and predict the slot label (column ``S2``). As you can see, each word in
the input gets assigned either an empty label ``O`` or a slot label that
begins with ``B-`` for the first word, and with ``I-`` for any
additional consecutive word that belongs to the same slot.

The model we will use is a recurrent model consisting of an embedding
layer, a recurrent LSTM cell, and a dense layer to compute the posterior
probabilities:

::

    slot label   "O"        "O"        "O"        "O"  "B-fromloc.city_name"
                  ^          ^          ^          ^          ^
                  |          |          |          |          |
              +-------+  +-------+  +-------+  +-------+  +-------+
              | Dense |  | Dense |  | Dense |  | Dense |  | Dense |  ...
              +-------+  +-------+  +-------+  +-------+  +-------+
                  ^          ^          ^          ^          ^
                  |          |          |          |          |
              +------+   +------+   +------+   +------+   +------+   
         0 -->| LSTM |-->| LSTM |-->| LSTM |-->| LSTM |-->| LSTM |-->...
              +------+   +------+   +------+   +------+   +------+   
                  ^          ^          ^          ^          ^
                  |          |          |          |          |
              +-------+  +-------+  +-------+  +-------+  +-------+
              | Embed |  | Embed |  | Embed |  | Embed |  | Embed |  ...
              +-------+  +-------+  +-------+  +-------+  +-------+
                  ^          ^          ^          ^          ^
                  |          |          |          |          |
    w      ------>+--------->+--------->+--------->+--------->+------... 
                 BOS      "show"    "flights"    "from"   "burbank"

Or, as a CNTK network description. Please have a quick look and match it
with the description above: (descriptions of these functions can be
found at: `the layers
reference <http://cntk.ai/pythondocs/layerref.html>`__

.. code:: python

    # number of words in vocab, slot labels, and intent labels
    vocab_size = 943 ; num_labels = 129 ; num_intents = 26    
    
    # model dimensions
    input_dim  = vocab_size
    label_dim  = num_labels
    emb_dim    = 150
    hidden_dim = 300
    
    def create_model():
        with default_options(initial_state=0.1):
            return Sequential([
                Embedding(emb_dim),
                Recurrence(LSTM(hidden_dim), go_backwards=False),
                Dense(num_labels)
            ])

Now we are ready to create a model and inspect it.

.. code:: python

    # peek
    model = create_model()
    print(len(model.layers))
    print(model.layers[0].E.shape)
    print(model.layers[2].b.value)


.. parsed-literal::

    3
    (-1, 150)
    [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.]
    

As you can see the attributes of the model are fully accessible from
Python. The model has 3 layers. The first layer is an embedding and you
can access its parameter ``E`` (where the embeddings are stored) like
any other attibute of a Python object. Its shape contains a ``-1`` which
indicates that this parameter is not fully specified yet. When we decide
what data we will run through this network (very shortly) the shape will
be the size of the input vocabulary. We also print the bias term in the
last layer. Bias terms are by default initialized to 0 (but there's also
a way to change that).

CNTK Configuration
------------------

To train and test a model in CNTK, we need to create a model and specify
how to read data and perform training and testing.

In order to train we need to specify:

-  how to read the data
-  the model function, its inputs, and outputs
-  hyper-parameters for the learner such as the learning rate

A Brief Look at Data and Data Reading
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We already looked at the data. But how do you generate this format? For
reading text, this tutorial uses the ``CNTKTextFormatReader``. It
expects the input data to be in a specific format, as described
`here <https://github.com/Microsoft/CNTK/wiki/CNTKTextFormat-Reader>`__.

For this tutorial, we created the corpora by two steps: \* convert the
raw data into a plain text file that contains of TAB-separated columns
of space-separated text. For example:

``BOS show flights from burbank to st. louis on monday EOS (TAB) flight (TAB) O O O O B-fromloc.city_name O B-toloc.city_name I-toloc.city_name O B-depart_date.day_name O``

This is meant to be compatible with the output of the ``paste`` command.
\* convert it to CNTK Text Format (CTF) with the following command:

``python [CNTK root]/Scripts/txt2ctf.py --map query.wl intent.wl slots.wl --annotated True --input atis.test.txt --output atis.test.ctf``
where the three ``.wl`` files give the vocabulary as plain text files,
one word per line.

In these CTF files, our columns are labeled ``S0``, ``S1``, and ``S2``.
These are connected to the actual network inputs by the corresponding
lines in the reader definition:

.. code:: python

    def create_reader(path, is_training):
        return MinibatchSource(CTFDeserializer(path, StreamDefs(
             query         = StreamDef(field='S0', shape=vocab_size,  is_sparse=True),
             intent_unused = StreamDef(field='S1', shape=num_intents, is_sparse=True),  
             slot_labels   = StreamDef(field='S2', shape=num_labels,  is_sparse=True)
         )), randomize=is_training, epoch_size = INFINITELY_REPEAT if is_training else FULL_DATA_SWEEP)

.. code:: python

    # peek
    reader = create_reader("atis.train.ctf", is_training=True)
    reader.streams.keys()

Trainer
~~~~~~~

We also must define the training criterion (loss function), and also an
error metric to track. Below we make extensive use of ``Placeholders``.
Remember that the code we have been writing is not actually executing
any heavy computation it is just specifying the function we want to
compute on data during training/testing. And in the same way that it is
convenient to have names for arguments when you write a regular function
in a programming language, it is convenient to have Placeholders that
refer to arguments (or local computations that need to be reused).
Eventually, some other code will replace these placeholders with other
known quantities in the same way that in a programming language the
function will be called with concrete values bound to its arguments.

.. code:: python

    def create_criterion_function(model):
        labels = Placeholder()
        ce   = cross_entropy_with_softmax(model, labels)
        errs = classification_error      (model, labels)
        return combine ([ce, errs]) # (features, labels) -> (loss, metric)

.. code:: python

    def train(reader, model, max_epochs=16):
        # criterion: (model args, labels) -> (loss, metric)
        #   here  (query, slot_labels) -> (ce, errs)
        criterion = create_criterion_function(model)
    
        criterion.replace_placeholders({criterion.placeholders[0]: Input(vocab_size),
                                        criterion.placeholders[1]: Input(num_labels)})
    
        # training config
        epoch_size = 18000        # 18000 samples is half the dataset size 
        minibatch_size = 70
        
        # LR schedule over epochs 
        # In CNTK, an epoch is how often we get out of the minibatch loop to
        # do other stuff (e.g. checkpointing, adjust learning rate, etc.)
        # (we don't run this many epochs, but if we did, these are good values)
        lr_per_sample = [0.003]*4+[0.0015]*24+[0.0003]
        lr_schedule = learning_rate_schedule(lr_per_sample, units=epoch_size)
        
        # Momentum (could also be on a schedule)
        momentum_as_time_constant = 700
        
        # We use a variant of the Adam optimizer which is known to work well on this dataset
        # Feel free to try other optimizers from 
        # https://www.cntk.ai/pythondocs/cntk.learner.html#module-cntk.learner
        learner = adam_sgd(criterion.parameters,
                           lr_per_sample=lr_schedule, momentum_time_constant=momentum_as_time_constant,
                           low_memory=True,
                           gradient_clipping_threshold_per_sample=15, gradient_clipping_with_truncation=True)
    
        # trainer
        trainer = Trainer(model, criterion.outputs[0], criterion.outputs[1], learner)
    
        # process minibatches and perform model training
        log_number_of_parameters(model)
        progress_printer = ProgressPrinter(tag='Training')
        #progress_printer = ProgressPrinter(freq=100, first=10, tag='Training') # more detailed logging
    
        t = 0
        for epoch in range(max_epochs):         # loop over epochs
            epoch_end = (epoch+1) * epoch_size
            while t < epoch_end:                # loop over minibatches on the epoch
                data = reader.next_minibatch(minibatch_size, input_map={  # fetch minibatch
                    criterion.arguments[0]: reader.streams.query,
                    criterion.arguments[1]: reader.streams.slot_labels
                })
                trainer.train_minibatch(data)                                     # update model with it
                t += data[criterion.arguments[1]].num_samples                     # samples so far
                progress_printer.update_with_trainer(trainer, with_metric=True)   # log progress
            loss, metric, actual_samples = progress_printer.epoch_summary(with_metric=True)
    
        return loss, metric

Running it
~~~~~~~~~~

You can find the complete recipe below.

.. code:: python

    def do_train():
        global model
        model = create_model()
        reader = create_reader("atis.train.ctf", is_training=True)
        train(reader, model)
    do_train()

This shows how learning proceeds over epochs (passes through the data).
For example, after four epochs, the loss, which is the cross-entropy
criterion, has reached 0.22 as measured on the ~18000 samples of this
epoch, and that the error rate is 5.0% on those same 18000 training
samples.

The epoch size is the number of samples--counted as *word tokens*, not
sentences--to process between model checkpoints.

Once the training has completed (a little less than 2 minutes on a
Titan-X or a Surface Book), you will see an output like this

::

    Finished Epoch [16]: [Training] loss = 0.058111 * 18014, metric = 1.3% * 18014

which is the loss (cross entropy) and the metric (classification error)
averaged over the final epoch.

On a CPU-only machine, it can be 4 or more times slower. You can try
setting

.. code:: python

    emb_dim    = 50 
    hidden_dim = 100

to reduce the time it takes to run on a CPU, but the model will not fit
as well as when the hidden and embedding dimension are larger.

Evaluating the model
~~~~~~~~~~~~~~~~~~~~

Like the train() function, we also define a function to measure accuracy
on a test set.

.. code:: python

    def evaluate(reader, model):
        criterion = create_criterion_function(model)
        criterion.replace_placeholders({criterion.placeholders[0]: Input(num_labels)})
    
        # process minibatches and perform evaluation
        dummy_learner = adam_sgd(criterion.parameters, 
                                 lr_per_sample=1, momentum_time_constant=0, low_memory=True)
        evaluator = Trainer(model, criterion.outputs[0], criterion.outputs[1], dummy_learner)
        progress_printer = ProgressPrinter(tag='Evaluation')
    
        while True:
            minibatch_size = 1000
            data = reader.next_minibatch(minibatch_size, input_map={  # fetch minibatch
                criterion.arguments[0]: reader.streams.query,
                criterion.arguments[1]: reader.streams.slot_labels
            })
            if not data:                                 # until we hit the end
                break
            metric = evaluator.test_minibatch(data)
            progress_printer.update(0, data[criterion.arguments[1]].num_samples, metric) # log progress
        loss, metric, actual_samples = progress_printer.epoch_summary(with_metric=True)
    
        return loss, metric

Now we can measure the model accuracy by going through all the examples
in the test set and using the ``test_minibatch`` method of the trainer
created inside the evaluate function defined above. At the moment (when
this tutorial was written) the Trainer constructor requires a learner
(even if it is only used to perform ``test_minibatch``) so we have to
specify a dummy learner. In the future it will be allowed to construct a
Trainer without specifying a learner as long as the trainer only calls
``test_minibatch``

.. code:: python

    def do_test():
        reader = create_reader("atis.test.ctf", is_training=False)
        evaluate(reader, model)
    do_test()
    model.layers[2].b.value

.. code:: python

    # load dictionaries
    query_wl = [line.rstrip('\n') for line in open('query.wl')]
    slots_wl = [line.rstrip('\n') for line in open('slots.wl')]
    query_dict = {query_wl[i]:i for i in range(len(query_wl))}
    slots_dict = {slots_wl[i]:i for i in range(len(slots_wl))}
    
    # let's run a sequence through
    seq = 'BOS flights from new york to seattle EOS'
    w = [query_dict[w] for w in seq.split()] # convert to word indices
    print(w)
    onehot = np.zeros([len(w),len(query_dict)], np.float32)
    for t in range(len(w)):
        onehot[t,w[t]] = 1
    pred = model.eval({model.arguments[0]:onehot})
    print(pred.shape)
    best = np.argmax(pred,axis=2)
    print(best[0])
    list(zip(seq.split(),[slots_wl[s] for s in best[0]]))

Modifying the Model
-------------------

In the following, you will be given tasks to practice modifying CNTK
configurations. The solutions are given at the end of this document...
but please try without!

A Word About ```Sequential()`` <https://www.cntk.ai/pythondocs/layerref.html#sequential>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before jumping to the tasks, let's have a look again at the model we
just ran. The model is described in what we call *function-composition
style*.

.. code:: python

            Sequential([
                Embedding(emb_dim),
                Recurrence(LSTM(hidden_dim), go_backwards=False),
                Dense(num_labels)
            ])

You may be familiar with the "sequential" notation from other
neural-network toolkits. If not,
```Sequential()`` <https://www.cntk.ai/pythondocs/layerref.html#sequential>`__
is a powerful operation that, in a nutshell, allows to compactly express
a very common situation in neural networks where an input is processed
by propagating it through a progression of layers. ``Sequential()``
takes an list of functions as its argument, and returns a *new* function
that invokes these functions in order, each time passing the output of
one to the next. For example,

.. code:: python

        FGH = Sequential ([F,G,H])
        y = FGH (x)

means the same as

::

        y = H(G(F(x))) 

This is known as `"function
composition" <https://en.wikipedia.org/wiki/Function_composition>`__,
and is especially convenient for expressing neural networks, which often
have this form:

::

         +-------+   +-------+   +-------+
    x -->|   F   |-->|   G   |-->|   H   |--> y
         +-------+   +-------+   +-------+

Coming back to our model at hand, the ``Sequential`` expression simply
says that our model has this form:

::

         +-----------+   +----------------+   +------------+
    x -->| Embedding |-->| Recurrent LSTM |-->| DenseLayer |--> y
         +-----------+   +----------------+   +------------+

Task 1: Add Batch Normalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We now want to add new layers to the model, specifically batch
normalization.

Batch normalization is a popular technique for speeding up convergence.
It is often used for image-processing setups, for example our other
`hands-on lab on image
recognition <./Hands-On-Labs-Image-Recognition>`__. But could it work
for recurrent models, too?

So your task will be to insert batch-normalization layers before and
after the recurrent LSTM layer. If you have completed the `hands-on labs
on image
processing <https://github.com/Microsoft/CNTK/blob/master/bindings/python/tutorials/CNTK_201B_CIFAR-10_ImageHandsOn.ipynb>`__,
you may remember that the `batch-normalization
layer <https://www.cntk.ai/pythondocs/layerref.html#batchnormalization-layernormalization-stabilizer>`__
has this form:

::

        BatchNormalization()

So please go ahead and modify the configuration and see what happens.

If everything went right, you will notice improved convergence speed
(``loss`` and ``metric``) compared to the previous configuration.

.. code:: python

    # Your task: Add batch normalization
    def create_model():
        with default_options(initial_state=0.1):
            return Sequential([
                Embedding(emb_dim),
                Recurrence(LSTM(hidden_dim), go_backwards=False),
                Dense(num_labels)
            ])
    
    do_train()
    do_test()

Task 2: Add a Lookahead
~~~~~~~~~~~~~~~~~~~~~~~

Our recurrent model suffers from a structural deficit: Since the
recurrence runs from left to right, the decision for a slot label has no
information about upcoming words. The model is a bit lopsided. Your task
will be to modify the model such that the input to the recurrence
consists not only of the current word, but also of the next one
(lookahead).

Your solution should be in function-composition style. Hence, you will
need to write a Python function that does the following:

-  takes no input arguments
-  creates a placeholder (sequence) variable
-  computes the "next value" in this sequence using the
   ``future_value()`` operation and
-  concatenates the current and the next value into a vector of twice
   the embedding dimension using ``splice()``

and then insert this function into ``Sequential()``'s list right after
the embedding layer.

.. code:: python

    # Your task: Add lookahead
    def create_model():
        with default_options(initial_state=0.1):
            return Sequential([
                Embedding(emb_dim),
                Recurrence(LSTM(hidden_dim), go_backwards=False),
                Dense(num_labels)
            ])
    do_train()
    do_test()

Task 3: Bidirectional Recurrent Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Aha, knowledge of future words help. So instead of a one-word lookahead,
why not look ahead until all the way to the end of the sentence, through
a backward recurrence? Let us create a bidirectional model!

Your task is to implement a new layer that performs both a forward and a
backward recursion over the data, and concatenates the output vectors.

Note, however, that this differs from the previous task in that the
bidirectional layer contains learnable model parameters. In
function-composition style, the pattern to implement a layer with model
parameters is to write a *factory function* that creates a *function
object*.

A function object, also known as
`*functor* <https://en.wikipedia.org/wiki/Function_object>`__, is an
object that is both a function and an object. Which means nothing else
that it contains data yet still can be invoked as if it was a function.

For example, ``Dense(outDim)`` is a factory function that returns a
function object that contains a weight matrix ``W``, a bias ``b``, and
another function to compute ``input @ W + b.`` (This is using `Python
3.5 notation for matrix
multiplication <https://docs.python.org/3/whatsnew/3.5.html#whatsnew-pep-465>`__.
In Numpy syntax it is ``input.dot(W) + b``). E.g. saying ``Dense(1024)``
will create this function object, which can then be used like any other
function, also immediately: ``Dense(1024)(x)``.

Let's look at an example for further clarity: Let us implement a new
layer that combines a linear layer with a subsequent batch
normalization. To allow function composition, the layer needs to be
realized as a factory function, which could look like this:

.. code:: python

    def DenseLayerWithBN(dim):
        F = Dense(dim)
        G = BatchNormalization()
        x = Placeholder()
        apply_x = G(F(x))
        return apply_x

Invoking this factory function will create ``F``, ``G``, ``x``, and
``apply_x``. In this example, ``F`` and ``G`` are function objects
themselves, and ``apply_x`` is the function to be applied to the data.
Thus, e.g. calling ``DenseLayerWithBN(1024)`` will create an object
containing a linear-layer function object called ``F``, a
batch-normalization function object ``G``, and ``apply_x`` which is the
function that implements the actual operation of this layer using ``F``
and ``G``. It will then return ``apply_x``. To the outside, ``apply_x``
looks and behaves like a function. Under the hood, however, ``apply_x``
retains access to its specific instances of ``F`` and ``G``.

Now back to our task at hand. You will now need to create a factory
function, very much like the example above. You shall create a factory
function that creates two recurrent layer instances (one forward, one
backward), and then defines an ``apply_x`` function which applies both
layer instances to the same ``x`` and concatenate the two results.

Allright, give it a try! To know how to realize a backward recursion in
CNTK, please take a hint from how the forward recursion is done. Please
also do the following: \* remove the one-word lookahead you added in the
previous task, which we aim to replace; and \* make sure each LSTM is
using ``hidden_dim//2`` outputs to keep the total number of model
parameters limited.

.. code:: python

    # Your task: Add bidirectional recurrence
    def create_model():
        with default_options(initial_state=0.1):  
            return Sequential([
                Embedding(emb_dim),
                Recurrence(LSTM(hidden_dim), go_backwards=False),
                Dense(num_labels)
            ])
    do_train()
    do_test()

Works like a charm! This model achieves 2.1%, a tiny bit better than the
lookahead model above. The bidirectional model has 40% less parameters
than the lookahead one. However, if you go back and look closely you may
find that the lookahead one trained about 30% faster. This is because
the lookahead model has both less horizontal dependencies (one instead
of two recurrences) and larger matrix products, and can thus achieve
higher parallelism.


Solution 1: Adding Batch Normalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    def create_model():
        with default_options(initial_state=0.1):
            return Sequential([
                Embedding(emb_dim),
                BatchNormalization(),
                Recurrence(LSTM(hidden_dim), go_backwards=False),
                BatchNormalization(),
                Dense(num_labels)
            ])
    
    do_train()
    do_test()

Solution 2: Add a Lookahead
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    def OneWordLookahead():
        x = Placeholder()
        apply_x = splice ([x, future_value(x)])
        return apply_x
    
    def create_model():
        with default_options(initial_state=0.1):
            return Sequential([
                Embedding(emb_dim),
                OneWordLookahead(),
                Recurrence(LSTM(hidden_dim), go_backwards=False),
                Dense(num_labels)        
            ])
    
    do_train()
    do_test()

Solution 3: Bidirectional Recurrent Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    def BiRecurrence(fwd, bwd):
        F = Recurrence(fwd)
        G = Recurrence(bwd, go_backwards=True)
        x = Placeholder()
        apply_x = splice ([F(x), G(x)])
        return apply_x 
    
    def create_model():
        with default_options(initial_state=0.1):
            return Sequential([
                Embedding(emb_dim),
                BiRecurrence(LSTM(hidden_dim//2), LSTM(hidden_dim//2)),
                Dense(num_labels)
            ])
    
    do_train()
    do_test()

