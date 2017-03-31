Layers Library Reference
========================

Note: This documentation has not yet been completely updated with respect to the latest update of the Layers library.
It should be correct but misses several new options and layer types.

CNTK predefines a number of common "layers," which makes it very easy to
write simple networks that consist of standard layers layered on top of
each other. Layers are function objects that can be used like a regular
:class:`~cntk.ops.functions.Function` but hold learnable parameters
and have an additional pair of ``()`` to pass construction parameters
or attributes.

For example, this is the network description for a simple 1-hidden layer
model using the :func:`~cntk.layers.layers.Dense` layer:

::

    h = Dense(1024, activation=relu)(features)
    p = Dense(9000, activation=softmax)(h)

which can then, e.g., be used for training against a cross-entropy
criterion:

::

    ce = cross_entropy(p, labels)

If your network is a straight concatenation of operations (many are),
you can use the alternative :ref:`sequential` notation:

::

    from cntk.layers import *
    my_model = Sequential ([
        Dense(1024, activation=relu),
        Dense(9000, activation=softmax)
    ])

and invoke it like this:

::

    p = my_model(features)

Built on top of ``Sequential()`` is :ref:`for`,
which allows to easily create models with repetitions. For example, a
2011-style feed-forward speech-recognition network with 6 hidden sigmoid
layers of identical dimensions can be written like this:

::

    my_model = Sequential ([
        For(range(6), lambda: \
            Dense(2048, activation=sigmoid))
        Dense(9000, activation=softmax)
    ])

Note that for most real-life inference scenarios, the output layer's
``softmax`` non-linearity is not needed (it is instead made part of the
training criterion).


General patterns
----------------

.. _specifying-the-same-options-to-multliple-layers:

Specifying the same options to multiple layers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Often, many layers share options. For example, typical image-recognition
systems use the ``relu`` activation function throughout. You can use the
Python ``with`` statement with the CNTK ``default_options()`` function
to define scopes with locally changed defaults, using one of the following two forms:

::

    with default_options(OPT1=VAL1, OPT2=VAL2, ...):
        # scope with modified defaults

    with default_options_for(FUNCTION, OPT1=VAL1, OPT2=VAL2, ...):
        # scope with modified defaults for FUNCTION only

The following options can be overridden with the ``with`` statement:

    - ``init`` (default: ``glorot_uniform()``): initializer specification, for :ref:`dense`, :ref:`convolution`, and :ref:`embedding`
    - ``activation`` (default: ``None``): activation function, for ``Dense()`` and ``Convolution()``
    - ``bias`` (default: ``True``): have a bias, for ``Dense()`` and ``Convolution()``
    - ``init_bias`` (default: ``0``): initializer specification for the bias, for ``Dense()`` and ``Convolution()``
    - ``initial_state`` (default: ``None``): initial state to use in ``Recurrence()`` :ref:`recurrence`
    - ``use_peepholes`` (default: ``False``): use peephole connections in ``LSTM()`` :ref:`lstm`

The second form allows to set default options on a
per-layer type. This is, for example, valuable for the ``pad``
parameter which enables padding in convolution and pooling, but is not
always set to the same for these two layer types.

Weight sharing
~~~~~~~~~~~~~~

If you assign a layer to a variable and use it in multiple places, *the
weight parameters will be shared*. If you say

::

    lay = Dense(1024, activation=sigmoid)
    h1 = lay(x)
    h2 = lay(h1)  # same weights as h1

``h1`` and ``h2`` will *share the same weight parameters*, as ``lay()``
is the *same function* in both cases. In the above case this is probably
not what was desired, so be aware. If both invocations of ``lay()``
above are meant to have different parameters, remember to define two
separate instances, for example ``lay1 = Dense(...)`` and
``lay2 = Dense(...)``.

So why this behavior? Layers allow to share parameters across sections
of a model. Consider a DSSM model which processes two input images, say
``doc`` and ``query`` identically with the same processing chain, and
compares the resulting hidden vectors:

::

    with default_options(activation=relu):
        image_to_vec = Sequential([
            Convolution((5,5), 32, pad=True), MaxPooling((3,3), strides=2),
            Convolution((5,5), 64, pad=True), MaxPooling((3,3), strides=2),
            Dense(64),
            Dense(10, activation=None)
        ])
    z_doc   = image_to_vec (doc)
    z_query = image_to_vec (query)  # same model as for z_doc
    sim = cos_distance(zdoc, z_query)

where ``image_to_vec`` is the part of the model that converts images
into flat vector. ``image_to_vec`` is a function object that in turn
contains several function objects (e.g. three instances of
``Convolution()``). ``image_to_vec`` is instantiated *once*, and this
instance holds the learnable parameters of all the included function
objects. Both invocations of ``model()`` will share these parameters in
application, and their gradients will be the sum of both invocations.

Lastly, note that if in the above example ``query`` and ``doc`` must
have the same dimensions, since they are processed through the same
function object, and that function object's first layer has its input
dimension inferred to match that of both ``query`` and ``doc``. If their
dimensions differ, then this network is malformed, and dimension
inference/validation will fail with an error message.

Example models
--------------

The following shows a slot tagger that embeds a word sequence, processes
it with a recurrent LSTM, and then classifies each word:

::

    from cntk.layers import *
    tagging_model = Sequential ([
        Embedding(150),         # embed into a 150-dimensional vector
        Recurrence(LSTM(300)),  # forward LSTM
        Dense(labelDim)         # word-wise classification
    ])

And the following is a simple convolutional network for image
recognition, using the
``with default_options(...):`` :ref:`specifying-the-same-options-to-multliple-layers`
pattern):

::

    with default_options(activation=relu):
        conv_net = Sequential ([
            # 3 layers of convolution and dimension reduction by pooling
            Convolution((5,5), 32, pad=True), MaxPooling((3,3), strides=2),
            Convolution((5,5), 32, pad=True), MaxPooling((3,3), strides=2),
            Convolution((5,5), 64, pad=True), MaxPooling((3,3), strides=2),
            # 2 dense layers for classification
            Dense(64),
            Dense(10, activation=None)
        ])

Notes
~~~~~

Many layers are wrappers around underlying CNTK primitives, along with
the respective required learnable parameters. For example,
```Convolution()`` :ref:`convolution` wraps the ``convolution()``
primitive. The benefits of using layers are: \* layers contain learnable
parameters of the correct dimension \* layers are composable (cf.
```Sequential()`` :ref:`sequential`)

However, since the layers themselves are implemented in Python using the
same CNTK primitives that are available to the user, if you find that a
layer you need is not available, you can always write it yourself or
write the formula directly as a CNTK expression.

The Python library described here is the equivalent of BrainScript's
`Layers Library <https://github.com/Microsoft/CNTK/wiki/BrainScript-Layers-Reference>`__.

.. _dense:

Dense()
-------

Factory function to create a fully-connected layer. ``Dense()`` takes an
optional activation function.

::

    Dense(shape, activation=default_override_or(identity), init=default_override_or(glorot_uniform()),
          input_rank=None, map_rank=None,
          bias=default_override_or(True), init_bias=default_override_or(0),
          name='')

Parameters
~~~~~~~~~~

-  ``shape``: output dimension of this layer
-  ``activation`` (default: ``None``: pass a function here to be used as
   the activation function, such as ``activation=relu``
-  ``input_rank``: if given, number of trailing dimensions that are
   transformed by ``Dense()`` (``map_rank`` must not be given)
-  ``map_rank``: if given, the number of leading dimensions that are not
   transformed by ``Dense()`` (``input_rank`` must not be given)
-  ``init`` (default: ``glorot_uniform()``): initializer descriptor for
   the weights. See :mod:`cntk.initializer`
   for a full list of random-initialization options.
-  ``bias``: if ``False``, do not include a bias parameter
-  ``init_bias`` (default: ``0``): initializer for the bias

Return Value
~~~~~~~~~~~~

A function that implements the desired fully-connected layer. See
description.

Description
~~~~~~~~~~~

Use these factory functions to create a fully-connected layer. It
creates a function object that contains a learnable weight matrix and,
unless ``bias=False``, a learnable bias. The function object can be used
like a function, which implements one of these formulas (using Python
3.5 ``@`` operator for matrix multiplication):

::

    Dense(...)(v) = activation (v @ W + b)
    Dense(...)(v) = v @ W + b      # if activation is None

where ``W`` is a weight matrix of dimension
``((dimension of v), shape)``, ``b`` is the bias of dimension
``(outdim,)``, and the resulting value has dimension (or tensor
dimensions) as given by ``shape``.

Tensor support
~~~~~~~~~~~~~~

If the returned function is applied to an input of a tensor rank > 1,
e.g. a 2D image, ``W`` will have the dimension
``(..., (second dimension of input), (first dimension of input), shape)``.

On the other hand, ``shape`` can be a vector that specifies tensor
dimensions, for example ``(10,10)``. In that case, ``W`` will have the
dimension ``((dimension of input), ..., shape[1], shape[0])``, and ``b``
will have the tensor dimensions ``(..., shape[1], shape[0])``.

CNTK's matrix product will interpret these extra output or input
dimensions as if they were flattened into a long vector. For more
details on this, see the documentation of
`Times() <https://github.com/Microsoft/CNTK/wiki/Times-and-TransposeTimes>`_.

The options ``input_rank`` and ``map_rank``, which are mutually
exclusive, can specify that not all of the input axes of a tensor should
be transformed. ``map_rank`` specifies how many leading axes are kept as
dimensions in the result; those axes are not part of the projection, but
rather each element along these axes is transformed independently (aka
*mapped*). ``input_rank`` is an alternative that instead specifies the
how many trailing axes are to be transformed (the remaining are mapped).

Example:
~~~~~~~~

::

    h = Dense(1024, activation=sigmoid)(v)

or alternatively:

::

    layer = Dense(1024, activation=sigmoid)
    h = layer(v)

.. _convolution:

Convolution()
-------------

Creates a convolution layer with optional non-linearity.

::

    Convolution(filter_shape,     # shape of receptive field, e.g. (3,3)
                num_filters=None, # e.g. 64 or None (which means 1 channel and don't add a dimension)
                sequential=False, # time convolution if True (filter_shape[0] corresponds to dynamic axis)
                activation=default_override_or(identity),
                init=default_override_or(glorot_uniform()),
                pad=default_override_or(False),
                strides=1,
                bias=default_override_or(True),
                init_bias=default_override_or(0),
                reduction_rank=1, # (0 means input has no depth dimension, e.g. audio signal or B&W image)
                max_temp_mem_size_in_samples=0,
                name='')

Parameters
~~~~~~~~~~

-  ``filter_shape``: shape of receptive field of the filter, e.g. ``(5,5)``
   for a 2D filter (not including the input feature-map depth)
-  ``num_filters``: number of output channels (number of filters)
-  ``activation``: optional non-linearity, e.g. ``activation=relu``
-  ``init``: initializer descriptor for the weights, e.g.
   ``glorot_uniform()``. See :mod:`cntk.initializer` for a full
   list of random-initialization options.
-  ``pad``: if ``False`` (default), then the filter will be shifted over
   the "valid" area of input, that is, no value outside the area is
   used. If ``pad`` is ``True`` on the other hand, the filter will be
   applied to all input positions, and values outside the valid region
   will be considered zero.
-  ``strides``: increment when sliding the filter over the input. E.g.
   ``(2,2)`` to reduce the dimensions by 2
-  ``bias``: if ``False``, do not include a bias parameter
-  ``init_bias``: initializer for the bias
-  ``use_correlation``: currently always ``True`` and cannot be changed.
   It indicates that ``Convolution()`` actually computes the
   cross-correlation rather than the true convolution

Return Value
~~~~~~~~~~~~

A function that implements the desired convolution operation.

Description
~~~~~~~~~~~

Use these factory functions to create a convolution layer.

The resulting layer applies a convolution operation on N-dimensional
feature maps. The caller specifies the receptive field of the filter and
the number of filters (output feature maps).

A set of filters for a given receptive field (e.g. ``(5,5)``) is
correlated with every location of the input (e.g. a ``(480, 640)``-sized
image). Assuming padding is enabled (``pad``) and strides are 1, this
will generate an output of the same dimension (``(480, 640)``).

Typically, many filters are applied at the same time, to create
"per-pixel activation vectors". ``num_filters`` specifies the number:
For every input location, an entire vector of ``num_filters`` is
produced. For our example above, setting ``num_filters`` to 64 would in
a ``(64, 480, 640)``-sized tensor. That first axis is also called the
*channel dimension* or the *feature-map axis*.

When convolution is applied to an input with a channel dimension, each
filter will also consist of vectors of the input's channel dimension.
E.g. when applying convolution with a specified receptive field of
``(5,5)`` to a ``(3, 480, 640)``-sized color image, each filter will be
a ``(3, 5, 5)]`` tensor.

All ``num_filters`` filters are stacked together into the so-called
convolution *kernel*, which is a parameter tensor owned by and held
inside this layer. In our example, the kernel shape will be
``(64, 3, 5, 5)``.

The following summarizes the relationship between the various dimensions
and shapes:

::

    input shape   : (               num_input_channels, (spatial dims) )
    filter shape  : (                                   (filter_shape) )
    output shape  : ( num_filters,                      (spatial dims) )
    kernel shape  : ( num_filters,  num_input_channels, (filter_shape)     )

which in our example are:

::

    input shape   : (              3, 480, 640 )
    filter shape  : (                   5, 5   )
    output shape  : ( num_filters,    480, 640 )
    kernel shape  : ( num_filters, 3,   5, 5   )

Padding
~~~~~~~

If padding is not enabled (``pad`` not given or ``False`` for all
dimensions), then the output size will be reduced by stripping the
boundary locations to which the full filter extent cannot be applied.
E.g. applying a ``(5,5)``-extent filter to an image without padding, the
outermost 2 rows and columns of pixels would cause the filter to be
applied out of bounds. Hence, ``Convolution()`` will reduce the
dimensions accordingly: An ``(480, 640)`` image convolved with a
``(5,5)`` filter without padding will leave a ``(476, 636)``-sized
output.

Strides
~~~~~~~

The ``strides`` parameters specify the increment of filters. Stride
values greater than one will lead to a sub-sampling of the output
region. E.g. filtering a ``(480, 640)`` image with a stride of ``(2,2)``
will result in a ``(240, 320)``-sized region with padding, and
``(238, 318)`` without padding.

Notes
~~~~~

This layer is a wrapper around the ``convolution()`` primitive.

The filter kernel parameters' name as shown in the log's validation
section will end in ``.W``.

Atrous convolution is at present not supported but planned for the near
future.

Example:
~~~~~~~~

::

    c = Convolution((3,3), 64, pad=True, strides=(1,1), bias=False)(x)

MaxPooling(), AveragePooling()
------------------------------

Factory functions to create a max- or average-pooling layer.

::

    MaxPooling(filter_shape,      # shape of receptive field, e.g. (3,3)
               strides=1,
               pad=default_override_or(False),
               name='')
    AveragePooling(filter_shape,  # shape of receptive field, e.g. (3,3)
                   strides=1,
                   pad=default_override_or(False),
                   name='')

Parameters
~~~~~~~~~~

-  ``filter_shape``: receptive field (window) to pool over, e.g. ``(2,2)``
   (not including the input feature-map depth)
-  ``strides``: increment when sliding the pool over the input. E.g.
   ``(2,2)`` to reduce the dimensions by 2
-  ``pad``: if ``False`` (default), then the pool will be shifted over
   the "valid" area of input, that is, no value outside the area is
   used. If ``pad`` is ``True`` on the other hand, the pool will be
   applied to all input positions, and values outside the valid region
   will be considered zero. For average pooling, count for average does
   not include padded values.

Return Value
~~~~~~~~~~~~

A function that implements the desired pooling layer.

Description
~~~~~~~~~~~

Use this factory function to create a pooling operation. Use
``MaxPooling()`` to compute the maximum over the values in the sliding
pooling window, and ``AveragePooling()`` to take their average.

The pooling operation slides a receptive field, or pooling window, over
the input, and computes either the maximum or the average of the values
in the respective window. In case of average with ``pad`` being
``True``, the padding regions are not included in the average.

This operation is structurally very similar to convolution, except that
the operation applied to the sliding window is of a different nature.

All considerations regarding input dimensions, padding, and strides
apply, so please see :ref:`convolution` for more
detail.

Example:
~~~~~~~~

::

    p = MaxPooling((3,3), strides=(2,2))(c)

GlobalMaxPooling(), GlobalAveragePooling()
------------------------------------------

Factory functions to create a global-max-pooling or global-average-pooling layer.

::

    GlobalMaxPooling(name='')
    GlobalAveragePooling(name='')

Return Value
~~~~~~~~~~~~

A function that implements the desired pooling layer.

Description
~~~~~~~~~~~

Use this factory function to create a global pooling operation. Use
``GlobalMaxPooling()`` to compute the maximum over all spatial data,
or ``GlobalAveragePooling()`` to take their average.

The global pooling operation infer the pooling window shape from the input
tensor and create a pooling function with pooling window size that
matches the input spatial dimension. It then computes either the
maximum or the average of all the values inside the inferred pooling
window.

Example:
~~~~~~~~

::

    p = GlobalMaxPooling()(c)

Dropout()
------------------------------

Factory functions to create a dropout layer.

::

    Dropout(dropout_rate=None, keep_prob=None, name='')

Parameters
~~~~~~~~~~

-  ``dropout_rate``: a fraction between [0, 1) that specifies the probability by which
   the dropout operation will randomly set elements of the input to zero. 0 mean
   select everything and close to 1 mean drop every element.

Return Value
~~~~~~~~~~~~

A function that implements the desired dropout layer.

Description
~~~~~~~~~~~

Use this factory function to create a dropout operation with a specific
dropout rate.

Example:
~~~~~~~~

::

    p = Dropout(0.5)(c)

.. _embedding:

Embedding()
-----------

Factory function to create a linear embedding layer, which is either
learned or a constant passed from outside.

::

    Embedding(shape=None, init=default_override_or(glorot_uniform()), weights=None, name='')

Parameters
~~~~~~~~~~

-  ``shape``: the dimension of the desired embedding vector. Must not be
   ``None`` unless ``weights`` are passed
-  ``init``: initializer descriptor for the weights to be learned. See
   :mod:`cntk.initializer` for a full
   list of initialization options.
-  ``weights`` (numpy array): if given, embeddings are not learned but
   specified by this array (which could be, e.g., loaded from a file)
   and not updated further during training

Return Value
~~~~~~~~~~~~

A function that implements the embedding layer. See description.

Description
~~~~~~~~~~~

"Embedding" refers to representing words or other discrete items by
dense continuous vectors. This layer assumes that the input is in
one-hot form. E.g., for a vocabulary size of 10,000, each input vector
is expected to have dimension 10,000 and consist of zeroes except for
one position that contains a 1. The index of that location is the index
of the word or item it represents.

In CNTK, the corresponding embedding vectors are stored as rows of a
matrix. Hence, mapping an input word to its embedding is implemented as
a matrix product. For this to be very efficient, it is important that
the input vectors are stored in sparse format (specify
``is_sparse=True`` in the corresponding ``Input()``).

Fun fact: The gradient of an embedding matrix has the form of gradient
vectors that are only non-zero for words seen in a minibatch. Since for
realistic vocabularies of tens or hundreds of thousands, the vast
majority of rows would be zero, CNTK implements a specific optimization
to represent the gradient in "row-sparse" form.

Known issue: The above-mentioned row-sparse gradient form is currently
not supported by our `1-bit
SGD <https://github.com/Microsoft/CNTK/wiki/Multiple-GPUs-and-machines#21-data-parallel-training-with-1-bit-sgd>`__
parallelization technique. Please use the
`block-momentum <https://github.com/Microsoft/CNTK/wiki/Multiple-GPUs-and-machines#22-block-momentum-sgd>`__
technique instead.

Example
~~~~~~~

A learned embedding that represents words from a vocabulary of 87636 as
a 300-dimensional vector:

::

    input = Input(87636, is_sparse=True)  # word sequence, as one-hot vector, sparse format
    embEn = Embedding(300)(input)         # embed word as a 300-dimensional continuous vector

In addition to ``is_sparse=True``, one would also typically read sparse
data from disk. Here is an example of reading sparse text input with the
`CNTKTextFormatReader <https://github.com/Microsoft/CNTK/wiki/BrainScript-CNTKTextFormat-Reader>`_:

::

    source = MinibatchSource(CTFDeserializer('en2fr.ctf', StreamDefs(
        input   = StreamDef(field='E', shape=87636, is_sparse=True),
        labels  = StreamDef(field='F', shape=98624, is_sparse=True)
    )))

If, instead, the embedding vectors already exist and should be loaded
from a file, it would look like this:

::

    input = Input(87636, is_sparse=True)   # word sequence, as one-hot vector, sparse format
    embEn = Embedding(300, weights=np.load_txt('embedding-en.txt'))(w) # embedding from disk

where the file ``'embedding-en.txt'`` is the name of a file that would
be expected to consist of 87,636 text rows, each of which consisting of
300 space-separated numbers.

.. _recurrence:

Recurrence()
------------

Factory function to create a single-layer or multi-layer recurrence.

::

    Recurrence(step_function, go_backwards=default_override_or(False), initial_state=default_override_or(0), return_full_state=False, name='')
    RecurrenceFrom(step_function, go_backwards=default_override_or(False), return_full_state=False, name='')
    Fold(folder_function, go_backwards=default_override_or(False), initial_state=default_override_or(0), return_full_state=False, name='')
    UnfoldFrom(generator_function, until_predicate=None, length_increase=1, name='')

Parameters
~~~~~~~~~~

-  ``step_function``: the ``Function`` to recur over, for example ``LSTM()``
-  ``go_backwards`` (optional): if ``True``, the recurrence is run
   backwards
-  ``initial_state`` (optional, default 0): initial value of the hidden
   variable that is recurred over. Currently, ``initial_state`` cannot
   have a dynamic axis.

Return Value
~~~~~~~~~~~~

``Recurrence()`` creates a function that implements the desired layer that recurrently applies a
model, such as an LSTM, to an input sequence. This layer maps an input
sequence to a sequence of hidden states of the same length.

Description
~~~~~~~~~~~

This implements the recurrence to be applied to an input sequence along
a dynamic axis. This operation automatically handles batches of
variable-length input sequences. The initial value(s) of the hidden
state variable(s) are 0 unless specified by ``initial_state``.

The ``step_function`` must be a CNTK Function that takes the previous state
and a new input, and outputs a new state.
State may consist of multiple variables (e.g. ``h`` and ``c`` in the case of the LSTM).

Applying this layer to an input sequence will return the sequence of the
hidden states of the ``Function`` to recur over (in case of an LSTM, the
LSTM's memory cell's value is not returned). The returned sequence has
the same length as the input. If only the last state is desired, as in
sequence-classification or some sequence-to-sequence scenarios, use
``Fold()`` instead of ``Recurrence()``.

Any function with such a signature can be used.
For example, ``Recurrence(plus, initial_value=0)`` is a layer that computes a cumulative sum over the input data,
while ``Fold(element_max)`` is a layer that performs max-pooling over a sequence.

To create a bidirectional model with ``Recurrence()``, use two layers,
one with ``go_backwards=True``, and ``splice()`` the two outputs
together.

``initial_state`` may have a dynamic batch axis. In that case,
the preferred pattern is ``RecurrentFrom()``, which creates a function
that takes the initial state as its first argument(s), followed by the inputs.

Example
~~~~~~~

A simple text classifier, which runs a word sequence through a
recurrence and then passes the *last* hidden state of the LSTM to a
softmax classifer, could have this form:

::

    w = Input(...)                          # word sequence (one-hot vectors)
    e = Embedding(150)(w)                   # embed as a 150-dimensional dense vector
    h = Recurrence(LSTM(300))(e)            # left-to-right LSTM with hidden and cell dim 300
    t = select_last(h)                      # extract last hidden state
    z = Dense(10000, activation=softmax)(t) # softmax classifier

To create a bidirectional one-layer LSTM (e.g. using half the hidden
dimension compared to above), use this:

::

    h_fwd = Recurrence(LSTM(150))(e)
    h_bwd = Recurrence(LSTM(150), go_backwards=True)(e)
    h = splice (h_fwd, h_bwd)

.. _lstm:

LSTM(), GRU(), RNNUnit()
------------------------

Factory functions to create a stateless LSTM/GRU/RNN ``Function``, typically for
use with ``Recurrence()``.

::

    LSTM(shape, cell_shape=None, activation=default_override_or(tanh), use_peepholes=default_override_or(False),
         init=default_override_or(glorot_uniform()), init_bias=default_override_or(0),
         enable_self_stabilization=default_override_or(False),
         name='')
    GRU(shape, cell_shape=None, activation=default_override_or(tanh),
        init=default_override_or(glorot_uniform()), init_bias=default_override_or(0),
        enable_self_stabilization=default_override_or(False),
        name='')
    RNNUnit(shape, cell_shape=None, activation=default_override_or(sigmoid),
            init=default_override_or(glorot_uniform()), init_bias=default_override_or(0),
            enable_self_stabilization=default_override_or(False),
            name='')

Parameters
~~~~~~~~~~

-  ``shape``: dimension of the output
-  ``cell_shape`` (optional): the dimension of the LSTM's cell. If
   ``None``, the cell shape is identical to ``shape``. If specified, an
   additional linear projection will be inserted to project from the
   cell dimension to the output shape.
-  ``use_peepholes`` (optional): if ``True``, then use peephole
   connections in the LSTM
-  ``init``: initializer descriptor for the weights. See :mod:`cntk.initializer`
   for a full list of initialization options.
-  ``enable_self_stabilization`` (optional): if ``True``, insert a
   ``Stabilizer()`` for the hidden state and cell

Return Value
~~~~~~~~~~~~

A ``Function`` that implements stateless Long-Short-Term-Memory, Gated Recurrent Unit, or
plain recurrent unit,
typically for use with the ``Recurrence()`` family of higher-order layers.

Description
~~~~~~~~~~~

This creates a ``Function`` object that implements the LSTM, GRU, or a RNN block. It returns
its current state, and takes the previous state as an additional input.
The function is stateless; i.e., it is *not* a recurrent LSTM layer. Use
``Recurrence()`` to turn this into a recurrent layer that is applied
along a dynamic axis.

Example
~~~~~~~

See ``Recurrence()``.

Delay()
-------

Factory function to create a layer that delays its input.

::

    Delay(T=1, initial_state=default_override_or(0), name='')

Parameters
~~~~~~~~~~

-  ``T``: the number of time steps to delay. To access future values,
   use a negative value
-  ``initial_state`` (optiona, default=0): value to use for the delayed
   frames at the boundaries

Return Value
~~~~~~~~~~~~

A function that implements the desired delay operation.

Description
~~~~~~~~~~~

This operation delays an input sequence by ``T`` steps (default 1). This
useful, for example, to turn a word sequence into a sequence of
overlapping word triples.

Consider an input sequence "a b c b", which shall be encoded as a
sequence of 3-dimensional one-hot vectors as follows:

::

    1 0 0
    0 1 0
    0 0 1
    0 1 0

Here, every row is a one-hot vector and corresponds to a word. Applying
``Delay(T=1)`` to this input will generate this sequence:

::

    0 0 0
    1 0 0
    0 1 0
    0 0 1

All tokens get delayed by one, and the first position gets filled in by
``initial_state`` which defaults to 0. Likewise, using ``Delay(T=-1)``
(negative delay) will give access to the future values, and pad from the
end with a zero:

::

    0 1 0
    0 0 1
    0 1 0
    0 0 0

Notes
~~~~~

This layer is a wrapper around the ``sequence.past_value()`` and
``sequence.future_value()`` primitives.

Example
~~~~~~~

The following shows how to stack three neighbor words into a trigram
vector:

::

    x  = ...                   # input value, e.g. a N-dimensional one-hot vector
    xp = Delay()(x)            # previous value
    xn = Delay(T=-1)(x)        # next value (negative delay)
    tg = splice (xp, x, xn)    # concatenate all into a 3N-dimensional three-hot vector

BatchNormalization(), LayerNormalization(), Stabilizer()
--------------------------------------------------------

Factory functions to create layers for batch normalization, layer
normalization, and self-stabilization.

::

    BatchNormalization(map_rank=default_override_or(None),  # if given then normalize only over this many dimensions. E.g. pass 1 to tie all (h,w) in a (C, H, W)-shaped input
                       init_scale=1,
                       normalization_time_constant=default_override_or(5000), blend_time_constant=0,
                       epsilon=default_override_or(0.00001), use_cntk_engine=default_override_or(False),
                       name='')
    LayerNormalization(initial_scale=1, initial_bias=0, epsilon=default_override_or(0.00001), name='')
    Stabilizer(steepness=4, enable_self_stabilization=default_override_or(True), name='')

Parameters
~~~~~~~~~~

``BatchNormalization``:

-  ``map_rank``: if given then normalize only over this many leading
   dimensions. E.g. 1 to tie all (h,w) in a (C, H, W)-shaped input.
   Currently, the only allowed values are ``None`` (no pooling) and
   ``1`` (e.g. pooling across all pixel positions of an image)
-  ``normalization_time_constant`` (default 5000): time constant in
   samples of the first-order low-pass filter that is used to compute
   mean/variance statistics for use in inference
-  ``initial_scale``: initial value of scale parameter
-  ``epsilon``: small value that gets added to the variance estimate
   when computing the inverse
-  ``use_cntk_engine``: if ``True``, use CNTK's native implementation.
   If false, use cuDNN's implementation (GPU only).

``LayerNormalization``:

-  ``initial_scale``: initial value of scale parameter
-  ``initial_bias``: initial value of bias parameter

``Stabilizer``:

-  ``steepness``: sharpness of the knee of the softplus function

Return Value
~~~~~~~~~~~~

A function that implements a layer that performs the normalization
operation.

Description
~~~~~~~~~~~

``BatchNormalization()`` implements the technique described in paper
`Batch Normalization: Accelerating Deep Network Training by Reducing
Internal Covariate Shift (Sergey Ioffe, Christian
Szegedy) <https://arxiv.org/abs/1502.03167>`__. It normalizes its inputs
for every minibatch by the minibatch mean/variance, and de-normalizes it
with a learned scaling factor and bias.

In inference, instead of using minibatch mean/variance, batch
normalization uses a long-term running mean/var estimate. This estimate
is computed during training by low-pass filtering minibatch statistics.
The time constant of the low-pass filter can be modified by the
``normalization_time_constant`` parameter. We recommend to start with
the default of (5000), but experiment with other values, typically on
the order of several thousand to tens of thousand.

Batch normalization currently requires a GPU for training.

``LayerNormalization()`` implements `Layer Normalization (Jimmy Lei Ba,
Jamie Ryan Kiros, Geoffrey E.
Hinton) <https://arxiv.org/abs/1607.06450>`__. It normalizes each input
sample by subtracting the mean across all elements of the sample, and
then dividing by the standard deviation over all elements of the sample.

``Stabilizer()`` implements a self-stabilizer per `Self-stabilized deep
neural network (P. Ghahremani, J.
Droppo) <http://ieeexplore.ieee.org/document/7472719/>`__. This simple
but effective technique multiplies its input with a learnable scalar
(but unlike layer normalization, it does not first normalize the input,
nor does it subtract a mean). Note that compared to the original paper,
which proposes a linear scalar ``beta`` or an exponential one
``Exp (beta)``, we found it beneficial to use a sharpened softplus
operation per the second author's suggestion, which avoids both negative
values and instability from the exponential.

Notes
~~~~~

``BatchNormalization()`` is a wrapper around the
``batch_normalization()`` primitive. ``LayerNormalization()`` and
``Stabilizer()`` are expressed directly in Python as a CNTK expression.

Example
~~~~~~~

A typical layer in a convolutional network with batch normalization:

::

    def my_convo_layer(x, depth, init):
        c = Convolution(depth, (5,5), pad=True, init=init)(x)
        b = BatchNormalization(map_rank=1)(c)
        r = relu(b)
        p = MaxPooling((3,3), strides=(2,2))(r)
        return p

.. _sequential:

Sequential()
------------

Composes an list of functions into a new function that calls these
functions one after another ("forward function composition").

::

    Sequential(layers, name='')

Parameters
~~~~~~~~~~

``layers``: a list of functions which may be layer instances or
single-argument primitives, e.g. ``[ LinearLayer(1024), sigmoid ]``

Return value
~~~~~~~~~~~~

This function returns another Function. That returned function takes one
argument, and returns the result of applying all given functions in
sequence to the input.

Description
~~~~~~~~~~~

``Sequential()`` is a powerful operation that allows to compactly
express a very common situation in neural networks where an input is
processed by propagating it through a progression of layers. You may be
familiar with it from other neural-network toolkits.

``Sequential()`` takes an array of functions as its argument, and
returns a *new* function that invokes these function in order, each time
passing the output of one to the next. Consider this example:

::

    FGH = Sequential ([F, G, H])
    y = FGH (x)

The ``FGH`` function defined above means the same as

::

    y = H(G(F(x)))

This is known as `function
composition <https://en.wikipedia.org/wiki/Function_composition>`_,
and is especially convenient for expressing neural networks, which often
have this form:

::

         +-------+   +-------+   +-------+
    x -->|   F   |-->|   G   |-->|   H   |--> y
         +-------+   +-------+   +-------+


which is perfectly expressed by ``Sequential ([F, G, H])``. (An even
shorter alternative way of writing it is ``(F >> G >> H)``.)

Lastly, please be aware that the following expression:

::

    layer1 = Dense(1024)
    layer2 = Dense(1024)
    z = Sequential([layer1, layer2])(x)

means something different from:

::

    layer = Dense(1024)
    z = Sequential([layer, layer])(x)

In the latter form, the same function *with the same shared set of
parameters* is applied twice (typically not desired), while in the
former, the two layers have separate sets of parameters.

Example
~~~~~~~

Standard 4-hidden layer feed-forward network as used in the earlier
deep-neural network work on speech recognition:

::

    my_model = Sequential ([
        Dense(2048, activation=sigmoid),  # four hidden layers
        Dense(2048, activation=sigmoid),
        Dense(2048, activation=sigmoid),
        Dense(2048, activation=sigmoid),
        Dense(9000, activation=softmax)   # note: last layer is a softmax
    )
    features = Input(40)
    p = my_model(features)

.. _for:

For()
-----

Repeats a layer multiple times.

::

    For(rng, constructor, name='')

Parameters
~~~~~~~~~~

-  ``N``: number of repetitions
-  ``constructor``: a lambda with 0 or 1 argument that creates the layer

Return value
~~~~~~~~~~~~

This function returns another Function. That returned function takes one
argument, and returns the result of applying the repeated layers to the
input, where each layer is a separate instance with a distinct set of
model parameters.

Description
~~~~~~~~~~~

``For()`` creates a sequential model by repeatedly executing a
*constructor lambda* passed to it; that is, you pass a Python function
that creates a layer, e.g. using the Python ``lambda`` syntax.

For example, creating a stack of 3 Dense layers of identical shape:

::

         +------------+   +------------+   +------------+
    x -->| Dense(128) |-->| Dense(128) |-->| Dense(128) |--> y
         +------------+   +------------+   +------------+

is as easy as:

::

    model = For(range(3), lambda: Dense(128))

Note that because you pass in a lambda for creating the layer, each
layer will be separately constructed. This is important, because this
ensures that all layers have their own distinct set of model parameters.

That constructor lambda can optionally take one parameter, the layer
counter. E.g. if the output dimension should double in each layer,

::

         +------------+   +------------+   +------------+
    x -->| Dense(128) |-->| Dense(256) |-->| Dense(512) |--> y
         +------------+   +------------+   +------------+

the one-parameter lambda form allows you to say this (notice the
``lambda i``, which defines a function that takes one parameter named
``i``):

::

    model = For(range(3), lambda i: Dense(128 * 2**i))

or this:

::

    dims = [128,256,512]
    model = For(range(3), lambda i: Dense(dims[i]))

Example
~~~~~~~

The following creates a 9-hidden-layer VGG-style model. VGG is a popular
architecture for image recognition:

::

    with default_options(activation=relu):
        model = Sequential([
            For(range(3), lambda i: [  # lambda with one parameter
                Convolution((3,3), [64,96,128][i], pad=True),  # depth depends on i
                Convolution((3,3), [64,96,128][i], pad=True),
                MaxPooling((3,3), strides=(2,2))
            ]),
            For(range(2), lambda : [   # lambda without parameter
                Dense(1024),
                Dropout(0.5)
            ]),
            Dense(num_classes, activation=None)
        ])

The resulting model will have this structure (read this from top to
bottom)

+------------------+
| VGG9             |
+------------------+
| input: image     |
+------------------+
|                  |
+------------------+
| conv3-64         |
+------------------+
| conv3-64         |
+------------------+
| max3             |
+------------------+
|                  |
+------------------+
| conv3-96         |
+------------------+
| conv3-96         |
+------------------+
| max3             |
+------------------+
|                  |
+------------------+
| conv3-128        |
+------------------+
| conv3-128        |
+------------------+
| max3             |
+------------------+
|                  |
+------------------+
| FC-1024          |
+------------------+
| dropout0.5       |
+------------------+
|                  |
+------------------+
| FC-1024          |
+------------------+
| dropout0.5       |
+------------------+
|                  |
+------------------+
| FC-10            |
+------------------+
|                  |
+------------------+
| output: object   |
+------------------+
