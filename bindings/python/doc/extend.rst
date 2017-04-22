Extending CNTK
==============

CNTK provides extension possibilities through
 - custom operators in pure Python as so-called 'user functions'
 - custom learning algorithms (like SGD or Adam) as 'user learners'
 - custom minibatch sources as 'user minibatch sources'

User defined functions
----------------------
Implementing a custom operator in pure Python is simple matter of

 - inheriting from :class:`~cntk.ops.functions.UserFunction`
 - implementing ``forward()`` and ``backward()``, whose signatures dependent on the number of inputs and outputs
 - specifying the outputs' shape, data type and dynamic axes in
   ``infer_outputs()``

In the simplest case, just only one input and output, ``forward()`` takes an
argument and returns a tuple of a state and the result. The state can be used to
pass data from the forward to the backward pass, but can be set to None if not
needed.

Let's consider the example of a sigmoid. This is just for demonstration purposes - for real
computation better use :func:`~cntk.ops.sigmoid`.

As the derivative of :math:`\textrm{sigmoid}(x)` is :math:`\textrm{sigmoid}(x) * (1-\textrm{sigmoid}(x))` we
pass the :math:`\textrm{sigmoid}(x)` value as the state variable, which is then later
fed into backward(). Note that one can pass any Python value (including
tuple, strings, etc.)::

    from cntk.ops.functions import UserFunction

    class MySigmoid(UserFunction):
        def __init__(self, arg, name='MySigmoid'):
            super(MySigmoid, self).__init__([arg], name=name)

        def forward(self, argument, device=None, outputs_to_retain=None):
            sigmoid_x = 1 / (1 + np.exp(-argument))
            return sigmoid_x, sigmoid_x

        def backward(self, state, root_gradients):
            sigmoid_x = state
            return root_gradients * sigmoid_x * (1 - sigmoid_x)

        def infer_outputs(self):
            return [output_variable(self.inputs[0].shape, self.inputs[0].dtype,
                self.inputs[0].dynamic_axes)]

This can now be used as a normal operator like::

    from cntk import user_function
    s = user_function(MySigmoid(prev_node))

Note that we cannot pass the `UserFunction` instance directly into the graph.
It is representing a primitive function, which we have to pass through
`user_function()`.

In case, the operator is initialized with multiple inputs, ``forward()`` 's
``argument`` will be a list of those inputs::

    class MyPlus(UserFunction):
        def __init__(self, arg1, arg2, name='f1'):
            super(MyPlus, self).__init__([arg1, arg2], name=name)

        def forward(self, arguments, device=None, outputs_to_retain=None):
            # No state needs to be passed to backward() so we just
            # pass None
            return None, arguments[0] + arguments[1]

        def backward(self, state, root_gradients):
            return root_gradients

        def infer_outputs(self):
            # We just pass the meta information of the first operand. For real
            # scenarios, one would want to calculate what the actual output's
            # result would actually look like (considering broadcasting, etc.).
            return [output_variable(self.inputs[0].shape, self.inputs[0].dtype, self.inputs[0].dynamic_axes)]

If the UserFunction has more than one input, ``backward()`` is invoked
with an additional ``variables`` argument, which contains a dictionary of
Variable to the gradient data, whose values have to be set with the proper
gradients. If the gradient is not to be propagated to a particular input,
the value for that input's gradient can be left None::

    def backward(self, state, root_gradients, variables):
        for var in variables:
            variables[var] = ... # compute the gradient for var

        # no return value since all the data is already in variables


In case, the operator shall output multiple outputs, the signature of forward
changes to::

   self.forward(args, outputs, device, outputs_to_retain):
       ...


which means that there is the additional dictionary ``outputs``, whose values
have to be set with the proper data.
In addition, ``root_gradient`` in ``backward()`` is a dictionary of Variable to the
root_gradient.

Using user functions for debugging
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is now easy to just plug user function nodes into the graph to support
debugging. For instance, the following operator::

    class LambdaFunc(UserFunction):
        def __init__(self,
                arg,
                when=lambda arg: True,
                execute=lambda arg: print(arg),
                name=''):
            self.when = when
            self.execute = execute

            super(LambdaFunc, self).__init__([arg], name=name)

        def infer_outputs(self):
            return [output_variable(self.inputs[0].shape, self.inputs[0].dtype, self.inputs[0].dynamic_axes)]

        def forward(self, argument, device=None, outputs_to_retain=None):
            if self.when(argument):
                self.execute(argument)

            return None, argument

        def backward(self, state, root_gradients):
            return root_gradients

can now be used to trigger certain actions when the data in the graph shows some
interesting behavior, for instance::

    import pdb
    import numpy as np
    # ... setting up the graph
    debug_node = LambdaFunc(node,
            when=lambda arg: np.var(arg)>1,
            execute=lambda arg: pdb.set_trace())
    # out = ... using user_function(debug_node) ...
    # ... training out

Now, if the variance of the input tensor exceeds 1, we will be put into
debugging mode and can start inspection.

User defined learners
---------------------
Implementing a custom learner in pure Python is accomplished by
 - creating a class that inherits from :class:`cntk.learners.UserLearner`
 - implementing its :meth:`~cntk.learners.UserLearner.update` method

Here is an example, how normal stochastic gradient descent would be
reimplemented in a naive way::

    from cntk.learner import UserLearner

    class MySgd(UserLearner):

        def __init__(self, parameters, lr_schedule):
            super(MySgd, self).__init__(parameters, lr_schedule)

        def update(self, gradient_values, training_sample_count, sweep_end):
            eta = self.learning_rate() / training_sample_count
            for p, g in gradient_values.items():
                new_p = p - eta * C.constant(g)
                p.set_value(new_p.eval(as_numpy=False).data())
            return True

The class ``MySgd`` could then be used as a normal learner, e.g.::

    # z, ce, pe = <your model, loss and evaluation functions>
    # lr_per_minibatch = <your learning rate specification>
    trainer = Trainer(z, (ce, pe), MySgd(z.parameters, lr_per_minibatch))

While this approach might be good enough as a one-off, it is not the fastest
possible UserLearner implementation. In every call, a complete CNTK graph is
created and then destructed (``new_p``). To speed up the parameter update, this
computation can be moved to the constructor:: 

    class MySgdFast(UserLearner):

        def __init__(self, parameters, lr_schedule):
            super(MySgdFast, self).__init__(parameters, lr_schedule, as_numpy=False)

            self.new_p = {}
            self.grad_input = {}

            self.sample_count_input = cntk.input((), name='count')

            lr = lr_schedule[0]  # assuming constant learning rate
            eta = lr / self.sample_count_input

            # we need one graph per parameter shape
            for param in parameters:
                p_shape = param.shape
                self.grad_input[p_shape] = cntk.input(p_shape)
                self.new_p[p_shape] = param - eta * self.grad_input[p_shape]

        def update(self, gradient_values, training_sample_count, sweep_end):
            for p, g in gradient_values.items():
                new_p = self.new_p[p.shape]
                grad_input = self.grad_input[p.shape]

                data = {
                        self.sample_count_input: np.asarray(training_sample_count),
                        grad_input: g
                        }
                result = new_p.eval(data, as_numpy=False)
                shape = result.shape

                # result has the shape of a complete minibatch, but contains
                # only one tensor, which we want to write to p. This means, we
                # have to slice off the leading dynamic axes.
                static_tensor = result.data.slice_view([0]*len(shape),
                                                       shape[2:])
                p.set_value(static_tensor)

            return True

With this implementation, we keep the costly NumPy conversion to a bare
minimum, while speeding up the update process considerably.

Before starting a new learner, though, please check out :mod:`cntk.learners`
whether your learner is already available.

User defined minibatch sources
------------------------------
In order to make use of CNTK's training session, one has to provide the input data as an
instance of :class:`~cntk.io.MinibatchSource`. Although :mod:`cntk.io` already provides means to read
image, text, and speech data, there might be the need (e.g. in distributed scnearios) to
roll out one's own custom minibatch
source. This is possible in pure Python as simple matter of

 - inheriting from :class:`~cntk.io.UserMinibatchSource` and
 - implementing the following methods

   - ``stream_infos()``: returns a list of :class:`~cntk.io.StreamInformation` instances that describe the streams the minibatch source is providing
   - ``next_minibatch()``: returns the next minibatch data as a dictionary of :class:`~cntk.io.StreamInformation` instance to the data (instance of :class:`~cntk.io.MinibatchData`, which basically wraps the data).

In the following example, we reimplement parts of the CNTKTextFormatReader to show how it
is done in an end-to-end manner. As we can see, the majority of the lines below is
scenario-specific code that deals with parsing, etc.::

    import numpy as np
    from cntk.io import UserMinibatchSource, StreamInformation, MinibatchData

    # Our toy test data contains two sequences. 'x' is a sparse representation of the
    # features (numbers representing the words in our training data). 'y' is the one-hot
    # label.
    MBDATA = r'''0	|x 560:1	|y 1 0 0 0 0
    0	|x 0:1
    0	|x 0:1
    1	|x 560:1	|y 0 1 0 0 0
    1	|x 0:1
    1	|x 0:1
    1	|x 424:1
    '''

    class MyDataSource(UserMinibatchSource):
        def __init__(self, f_dim, l_dim):
            self.f_dim, self.l_dim = f_dim, l_dim

            self.fsi = StreamInformation("features", 0, 'sparse', np.float32, (self.f_dim,))
            self.lsi = StreamInformation("labels", 1, 'dense', np.float32, (self.l_dim,))

            # MBDATA fits into memory, so we will read it in all at once. Normally, however,
            # it does not, in which case we would need to keep track of the position in the
            # file until which we have already provided the data.
            # It follows the CNTKTextFormat specification
            #   sequence ID |feature1 data |feature2 data
            # where in this case feature1's data is encoded as one-hot and we will
            # convert to CSR, and feature2's data is a one-hot encoded as dense.

            # We will store
            #   sequence id -> "features" -> list of features
            # and
            #   sequence id -> "labels" -> label

            self.data = {}
            for line in MBDATA.split('\n'):
                line = line.strip()
                if not line:
                    continue
                seq_id, data = line.split('|', 1)
                data = data.split("|")
                seq_id = int(seq_id.strip())

                if seq_id not in self.data:
                    self.data[seq_id] = {'features': []}

                # Processing features - expecting one per line.
                features = data[0].split(" ")
                vocab_idx = int(features[1].split(":")[0])
                self.data[seq_id]['features'].append(vocab_idx)

                # Process label, if exists
                if len(data) == 2:
                    labels = np.asarray([data[1].split(" ")[1:]], dtype=np.float32)
                    self.data[seq_id]['labels'] = labels

            self.sequences = sorted(self.data)
            self.next_seq_idx = 0

            super(MyDataSource, self).__init__()

        def stream_infos(self):
            return [self.fsi, self.lsi]

        def next_minibatch(self, num_samples, number_of_workers=1, worker_rank=0, device=None):
            # Note that in this example we do not yet make use of number_of_workers or
            # worker_rank, which will limit the minibatch source to single GPU / single node
            # scenarios.

            features = []
            labels = []

            sweep_end = False

            f_sample_count = l_sample_count = 0

            while max(f_sample_count, l_sample_count) < num_samples:
                if self.next_seq_idx == len(self.sequences):
                    sweep_end = True
                    self.next_seq_idx = 0

                seq_id = self.sequences[self.sequences[self.next_seq_idx]]

                f_data = self.data[seq_id]['features']
                l_data = self.data[seq_id]['labels']
                if (features or labels) and max(f_sample_count+len(f_data), l_sample_count+len(l_data)) > num_samples:
                    break
                f_sample_count += len(f_data)
                features.append(f_data)

                l_sample_count += len(l_data)
                labels.append(l_data)

                self.next_seq_idx += 1

            num_seq = len(features)

            f_data = Value.one_hot(batch=features, num_classes=self.f_dim)
            l_data = Value(batch=np.asarray(labels, dtype=np.float32))

            result = {
                    self.fsi: MinibatchData(f_data, num_seq, f_sample_count, sweep_end),
                    self.lsi: MinibatchData(l_data, num_seq, l_sample_count, sweep_end)
                    }


            return result

This can then be used wherever a :class:`~cntk.io.MinibatchSource` instance is accepted,
e.g.::

    input_dim = 1000
    num_output_classes = 5

    # instantiating the user minibatch source
    mbs = MyDataSource(input_dim, num_output_classes)
    feature = sequence.input(shape=(input_dim,))
    label = input(shape=(num_output_classes,))

    # setting up the model
    # ...

    # and train
    trainer = Trainer(z, (ce, errs), [learner])
    input_map = {
        feature: mbs.fsi,
        label: mbs.lsi
    }

    session = training_session(
        trainer=trainer, mb_source=mbs,
        model_inputs_to_streams=input_map,
        mb_size=4, max_samples=20
    )
    session.train()

As we have noted above, this minibatch source is limited to single GPU / single node
scenarios, but it can be adapted easily to be used with e.g. BlockMomentum. We simply have
to use `number_of_workers` to cut the data in slices and then return the slices depending
on which `worker_rank` requested the next minibatch.

.. note:: Please note that it is the user's task to provide proper randomization of the training data.
