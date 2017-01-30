Extending CNTK
==============

CNTK allows to implement custom operators in pure Python as so-called 'user
functions'.


User functions
==============

Implement a custom operator in pure Python is simple matter of

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

    s = MySigmoid(prev_node)

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

If more than one input requires gradient calculation, ``backward()`` is invoked
with an additional ``variables`` argument, which contains a dictionary of
Variable to the gradient data, whose values have to be set with the proper
gradients::

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
----------------------------------

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
    # out = ... using debug_node ...
    # ... training out

Now, if the variance of the input tensor exceeds 1, we will be put into
debugging mode and can start inspection.

