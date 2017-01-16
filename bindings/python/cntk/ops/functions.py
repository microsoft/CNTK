from cntk import cntk_py
from cntk.device import DeviceDescriptor
from cntk.utils import typemap, sanitize_var_map, sanitize_batch, \
        sanitize_dtype_cntk, value_to_seq
from cntk.utils.swig_helper import map_if_possible
from enum import Enum, unique
import numpy as np


@unique
class CloneMethod(Enum):
    '''
    Describes different ways how :func:`~cntk.ops.functions.Function.clone`
    works.
    '''

    share = 'share'
    '''
    Parameters are shared between the Function being cloned and the new clone
    '''

    clone = 'clone'
    '''
    New learnable parameters are created and initialied with the current values of the
    corresponding parameters of the Function being cloned
    '''

    freeze = 'freeze'
    '''
    Parameters are cloned and made immutable; i.e. Constants in the new clone
    (e.g. for use as a fixed feature extractor)
    '''

class Function(cntk_py.Function):
    '''
    Base class of all primitive tensor operators.

    If it has only one output, one can invoke Variable methods on it, which it
    will relay to its only output.

    For all available methods, see :class:`Function`.
    '''

    # define input shapes, in-place
    # e.g.
    # model.declare_args(42)
    # pass a list of objects that define the dimensions etc. of the placeholders
    # Currently you can pass either
    def declare_args(self, *arg_types):
        placeholders = self.placeholders  # the unbound parameters to fill in
        if len(arg_types) != len(placeholders):
            raise TypeError("CNTK Function.declare_inputs() expected {} arguments, got {}".format(len(placeholders), len(arg_types)))
        def to_input(arg):
            if isinstance(arg, cntk_py.Variable):
                return arg
            else:
                from cntk import input_variable
                return input_variable(arg)
        args = [to_input(arg) for arg in arg_types]
        self.replace_placeholders(dict(zip(placeholders, args)))


    # call a function, i.e. clone with all placeholders/inputs replaced
    def __call__(self, *args):
        if not isinstance(args, tuple):  # normalize single argument into tuple
            args = (args,)
        # flatten args to a list. Note it may be a a tuple or even a nested tree of tuples, e.g. LSTM (x, (h, c))
        def flatten_tuple(args):
            if not isinstance(args, tuple): # not a tuple: singleton; create a singleton tuple
                return (args,)
            from operator import add
            from functools import reduce
            return reduce(add, [(flatten_tuple(item)) for item in args])
        args = list(flatten_tuple(args))  # normalize nested arg tuples into flat tuple  --TODO: is there a standard function to do this?
        # TODO: This should not be necessary, or go into Function.replace_placeholders()
        def _output_of(arg):  # helper to get the output of an arg; use arg itself if no output() method (that'd be a Variable)
            try:
                return arg.output
            except AttributeError:
                return arg  # Variables have no output()
        args = [_output_of(arg) for arg in args]  # normalize args to their outputs  --BUGBUG: without: "TypeError: cannot convert value of dictionary to CNTK::Variable "
        #from cntk.ops import combine
        #args = [combine([arg]) for arg in args]  # BUGBUG: without: "TypeError: cannot convert value of dictionary to CNTK::Variable "
        placeholders = self.placeholders  # the unbound parameters to fill in
        if len(args) != len(placeholders):
            raise TypeError("CNTK Function expected {} arguments, got {}".format(len(placeholders), len(args)))
        return self.clone(CloneMethod.share, dict(zip(placeholders, args)))

    # forward function composition (other o self)
    def __rshift__(self, other):
        return other(self)

    # backward function composition (self o other)
    def __lshift__(self, other):
        return self(other)

    def __getattr__(self, name):
        try:
            return self.__dict__[name]
        except KeyError:
            # If name is a member of self's single output, then we relay to
            # that.
            if name in ['outputs', 'output', 'this']:
                # 'outputs' and 'output' are required to fetch the attribute for 
                # in the Variable.
                # 'this' is required for Swig and needs to be thrown if the
                # object is created the first time.
                # All others we try to find in self.output.
                raise

            if len(self.outputs) == 1 and hasattr(self.output, name):
                return getattr(self.output, name)
            else:
                raise AttributeError("'%s' object has no attribute '%s'" %
                        (type(self), name))


    @property
    @typemap
    def arguments(self):
        '''
        List of all input variables of the Function that are not of type Parameter or Constant
        '''
        return super(Function, self).arguments()

    @property
    @typemap
    def attributes(self):
        '''
        List of the attributes of the function
        '''
        return super(Function, self).attributes()

    @typemap
    def clone(self, method, substitutions=None):
        '''
        Clones the function. The parameters of the Function are either cloned,
        shared or frozen as specified by the method argument and any variable
        substitutions requested are applied in the cloned Function instance.

        Args:
            method (:class:`CloneMethod`): one of

             * 'clone': the returned function gets its own copy of parameters (default)
             * 'share': the returned function shares its parameters with this function
             * 'freeze': parameters are cloned and made immutable (constant).

            substitutions (dict): a dictionary mapping variables in this
             function to variables in the cloned function

        Returns:
            :class:`~cntk.ops.functions.Function`: the cloned Function
        '''
        method = getattr(cntk_py,
                'ParameterCloningMethod_' + CloneMethod(method).name.capitalize())
        if substitutions is None:
            substitutions = {}
        return super(Function, self).clone(method, substitutions)

    @property
    @typemap
    def constants(self):
        '''
        List of all `Constant` variables of this :class:`~cntk.ops.functions.Function`
        '''
        return super(Function, self).constants()

    def eval(self, arguments=None, device=None):
        '''
        Evaluate the node using the specified ``arguments`` as input.

        Args:
            arguments: maps variables to their input data. The interpretation depends on
             the input type:

               * dict: keys are input variable or names, and values are the input data.
                 See :meth:`~cntk.ops.functions.Function.forward` for details on passing
                 input data.
               * any other type: if node has an unique input, arguments is
                 mapped to this input.
             For nodes with more than one input, only dict is allowed.

             In both cases, every every sample in the data will be interpreted
             as a new sequence.

             Sequences can be marked as continuations of the same sequence in
             the previous minibatch (that is the sequence in the same slot).
             There are two possibilities for this:

              * specifying arguments as a `tuple` where the first element is
                used as arguments and the second one will be used as a list
                of bools, denoting whether a sequence is a new one (`True`) or a
                continuation of the sequence in the same slot of the previous
                minibatch (`False`). This will be applied to all batches.
              * specifying arguments as a dictionary of variables to tuples
                where the first element is used as arguments and the second
                one will be used as a list of bools, denoting whether a sequence
                is a new one (`True`) or a continuation of the sequence in the
                same slot of the previous minibatch (`False`). This will be
                applied to all batches.

             Data should be either NumPy arrays or a
             :class:`~cntk.io.MinibatchData` instance.
            device (:class:`~cntk.device.DeviceDescriptor`): the device descriptor that
             contains the type and id of the device on which the computation is
             to be performed.

        Returns:
           dict or NumPy Array: Dict with keys of ouput variable names and values of
           output variable. A single NumPy array if there is only one output value.
        '''

        _, output_map = self.forward(arguments, self.outputs, device=device)

        if len(output_map) > 1:
            return output_map
        else:
            return list(output_map.values())[0]

    @typemap
    def forward(self, arguments, outputs, keep_for_backward=None, device=None):
        '''
        Computes the values of speficied variables in ``outputs``, using values
        provided in ``arguments`` that correspond to each input `Variable` of
        the function whose ``is_input`` is `True`.

        Example:
            >>> v = C.input_variable(shape=(3,))
            >>> f = C.reciprocal(v)
            >>> _, fv = f.forward({v:[[1, 2, 4]]}, [f.output])
            >>> list(fv.values())[0]
            array([[[ 1.  ,  0.5 ,  0.25]]], dtype=float32)

        Args:
            arguments: maps variables to their input data. The interpretation depends on
             the input type:

               * dict: keys are input variable or names, and values are the
                 input data. To specify a minibatch, provide a list of arrays.
                 The shape of each array must be compatible with the shape of
                 the dictionary key.If the array denotes a sequence then the
                 elements of the sequence are grouped along axis 0.
               * any other type: if node has an unique input, arguments is
                 mapped to this input.
             For nodes with more than one input, only dict is allowed.

             In both cases, every every sample in the data will be interpreted
             as a new sequence.

             Sequences can be marked as continuations of the same sequence in
             the previous minibatch (that is the sequence in the same slot).
             There are two possibilities for this:

              * specifying arguments as a `tuple` where the first element is
                used as arguments and the second one will be used as a list
                of bools, denoting whether a sequence is a new one (`True`) or a
                continuation of the sequence in the same slot of the previous
                minibatch (`False`). This will be applied to all batches.
              * specifying arguments as a dictionary of variables to tuples
                where the first element is used as arguments and the second
                one will be used as a list of bools, denoting whether a sequence
                is a new one (`True`) or a continuation of the sequence in the
                same slot of the previous minibatch (`False`). This will be
                applied to all batches.

             Data should be either NumPy arrays or a
             :class:`~cntk.io.MinibatchData` instance.
            outputs (iterable): outputs to fetch values for.
            keep_for_backward (set, default `None`): the subset of the
             Function's output variables for which gradients shall be calculated
             in a subsequent backward call. If `None`, the returned state will
             be `None` and a subsequent call to :func:`backward` will not be
             possible.
            device (:class:`~cntk.device.DeviceDescriptor`, default `None`): the device
             descriptor that contains the type and id of the device on which the
             computation is. If `None`, the default device is used.

        Returns:
             A tuple (BackpropState, map of outputs to NumPy arrays). The
             BackpropState is a handle taken by :func:`backward`.
        '''
        if device is None:
            device = DeviceDescriptor.use_default_device()

        in_var_map = sanitize_var_map(self.arguments, arguments,
                                      None, device)
        output_map = {v: None for v in outputs}
        keep_for_backward = set(keep_for_backward or {})

        state = super(Function, self)._forward(in_var_map, output_map, device,
                                             keep_for_backward)

        for k in output_map:
            output_map[k] = value_to_seq(output_map[k])

        return state, output_map

    @typemap
    def backward(self, state, root_gradients, variables):
        '''
        Backpropagates supplied ``root_gradients`` for one or more of the output
        variables of the Function, to calculate gradients with respect to
        ``variables``. Formally, multiplies the values of ``root_gradients`` by
        the Jacobian of the Function and returns the subset of the output that
        corresponds to ``variables``.

        Example:
            >>> # compute the value and the derivative of the sigmoid at 0
            >>> v = C.input_variable(shape=(1,), needs_gradient=True)
            >>> f = C.sigmoid(v)
            >>> df, fv = f.forward({v:[[0]]}, [f.output], set([f.output]))
            >>> value = list(fv.values())[0]
            >>> grad = f.backward(df, {f.output: np.ones_like(value)}, set([v]))
            >>> value
            array([[[ 0.5]]], dtype=float32)
            >>> list(grad.values())[0]
            array([[[ 0.25]]], dtype=float32)

        Args:
            state (BackPropState): state obtained from a previous call to the
             func:`cntk.ops.Function.forward` method on this Function for the
             computation that this gradient backpropagation corresponds to.
            root_gradients (dict): the gradients that will be backpropagated
            variables (set): a list of input variables with respect to which
             the gradients have to be computed.

        Returns:
            dict: mapping of ``variables`` to NumPy arrays
        '''
        device = state.device()
        root_gradients = sanitize_var_map(self.outputs, root_gradients,
                                          None, device)

        var_gradients = dict((var, None) for var in variables)

        self._backward(state, root_gradients, var_gradients)

        for var, value in var_gradients.items():
            var_gradients[var] = value_to_seq(value)

        return var_gradients

    @typemap
    def grad(self, at, wrt=None, device=None):
        '''
        Computes the gradient of this Function at location ``at`` with respect to ``wrt``.
        The Function must have a single output.

        Example:
            >>> x = C.input_variable(shape=(1,), needs_gradient=True)
            >>> y = C.sqrt(x)
            >>> a = np.asarray([1,4,16],dtype=np.float32).reshape(3,1,1)
            >>> y.grad({x:a})
            [array([[[ 0.5  ]],
            <BLANKLINE>
                   [[ 0.25 ]],
            <BLANKLINE>
                   [[ 0.125]]], dtype=float32)]

        Args:
            at (dict) : mapping of the Function's arguments to values
            wrt (list optional): list of Variables with respect to which the
             gradient will be computed. If omitted, the gradients with
             respect to all arguments will be computed. If a variable
             is repeated in this list, the gradient will be repeated
             in the output as a shallow copy.

        Returns:
            list: list containing the gradients in the same order as
            the variables in ``wrt``. Each element has the same shape as
            ``wrt`` including dynamic axes (such as the minibatch axis).
        '''

        if len(self.outputs) != 1 :
            raise InvalidArgumentException('function must return a single tensor')

        if wrt is None:
            wrt = self.arguments

        unique_wrt = set(wrt)
        output = [self.output]
        state, results = self.forward(at, output, set(output), device)
        ones = {self.output: np.ones_like(v) for v in results.values()}
        grad_dict = self.backward(state, ones, unique_wrt)
        return [grad_dict[v] for v in wrt]

    @property
    @typemap
    def inputs(self):
        '''
        List of all input variables of this function.
        '''
        return super(Function, self).inputs()

    @property
    def name(self):
        '''
        Name of this function
        '''
        return super(Function, self).name()

    @name.setter
    def name(self, function_name):
        '''
        Sets the name of this Function.
        Setting the name of a Function is only allowed if the Function does not already have a name.
        Calling this method, when this Function already has a name, results in an exception.

        Args:
            function_name (`str`): name for this Function.
        '''
        super(Function, self).set_name(function_name)

    @property
    def op_name(self):
        '''
        Name of the operation that this Function performs
        '''
        return super(Function, self).op_name()

    @property
    @typemap
    def output(self):
        '''
        The single output variable if there is only one, or raises an exception.
        '''
        return super(Function, self).output()

    @property
    @typemap
    def outputs(self):
        '''
        List consisting of all output variables of this function.
        '''
        return super(Function, self).outputs()

    @property
    @typemap
    def parameters(self):
        '''
        List of all parameter variables of this function.
        '''
        return super(Function, self).parameters()

    @property
    @typemap
    def placeholders(self):
        '''
        List of all placeholders variables of this function.
        '''
        return super(Function, self).placeholders()

    @property
    @typemap
    def root_function(self):
        '''
        The primitive function at the root of the graph of functions underlying this function.
        '''
        return super(Function, self).root_function()

    @property
    def is_primitive(self):
        '''
        Returns a boolean indicating if this Function is a primitive Function.
        A primitive Function is the lowest level building block for composite Function
        graphs and is either a CNTK built-in operator, a composite Function encapsulated
        as a Block or a user-defined Function
        '''
        return super(Function, self).is_primitive()

    @property
    def is_composite(self):
        '''
        Returns a boolean indicating if this Function is a composite Function.
        A composite Function is a Function that is composed of primitive Functions.
        '''
        return super(Function, self).is_composite()

    @property
    def is_block(self):
        '''
        Returns a boolean indicating if this Function is a block function which is basically
        a composite encapsulated as an opaque block which appears as a primitive during
        traversing the graph of Functions that this block is part of.
        '''
        return super(Function, self).is_block()

    @property
    @typemap
    def block_root(self):
        '''
        Returns the root of the Function graph underlying this block Function.
        Throws an exception of this is not a block Function.
        '''
        return super(Function, self).block_root()

    @property
    @typemap
    def block_arguments_mapping(self):
        '''
        Returns the mapping from the arguments of the composite underlying this block function
        to the Variables that they are bound to in the outer graph of Functions that this
        block Function is part of.
        '''
        return super(Function, self).block_arguments_mapping()

    @property
    @typemap
    def uid(self):
        '''
        The internally generated unique name of the function.
        '''
        return super(Function, self).uid()

    @typemap
    def replace_placeholders(self, substitutions):
        '''
        In-place replace specified placeholders in the Function graph with the
        specified replacements in the map.

        Args:
            substitutions (dict): map from placeholder to variables

        Returns:
            :class:`Function`: itself
        '''
        return super(Function, self).replace_placeholders(substitutions)

    @typemap
    def replace_placeholder(self, substitution):
        '''
        In-place replace the only placeholder in the function graph with the
        specified substitution.

        Args:
            substitution (:class:`~cntk.ops.variables.Variable`): the variable
             that will replace the placeholder

        Returns:
            :class:`Function`: itself

        :raises ExceptionType: when the function has multiple placeholders.
        '''
        return super(Function, self).replace_placeholder(substitution)

    @typemap
    def find_all_with_name(self, name):
        '''
        Returns a list of primitive function with ``name`` in the graph
        starting from this node. Throws an exception if ``name`` occurs
        multiple times. If you expect only one function to be returned, use
        :func:`find_by_name`.

        Example:
            >>> a = C.input_variable(shape=1, name='i')
            >>> b = C.input_variable(shape=1, name='i')
            >>> c = C.plus(a, b, name='c')
            >>> len(c.find_all_with_name('i'))
            2
            >>> c.find_all_with_name('z')
            []

        Args:
            name (str): names to look for

        Returns:
            list of :class:`Function` objects matching ``name``

        See also:
            :func:`find_by_name`
        '''
        from .. import graph
        return graph.find_all_with_name(self, name)

    # TODO have a better name for combine() in this case
    @typemap
    def find_by_name(self, name):
        '''
        Returns a primitive function with ``name`` in the graph starting from
        this node. Throws an exception if ``name`` occurs multiple times. If
        you expect multiple functions to be returned, use
        :func:`find_all_with_name`.

        Example:
            >>> a = C.input_variable(shape=1, name='a')
            >>> b = C.input_variable(shape=1, name='b')
            >>> c = C.plus(a, b, name='c')
            >>> print(c.find_by_name('b').name)
            b
            >>> c.find_by_name('z') is None
            True

            If you need a full function out of it that can be evaluated, you
            need to upcast it (currently done via combine):

            >>> d = c * 5
            >>> C.combine([d.find_by_name('c')]).eval({a:[[1]], b:[[2]]})
            array([[[ 3.]]], dtype=float32)

        Args:
            name (str): names to look for

        Returns:
            :class:`Function` object matching ``name``

        See also:
            :func:`find_all_with_name`
        '''
        from .. import graph
        return graph.find_by_name(self, name)

    @typemap
    def save_model(self, filename):
        '''
        Save this function graph into a model file using protobuf-based
        serialization.

        Args:
            filename (str): model path
        '''
        return super(Function, self).save_model(filename)

    @typemap
    def restore_model(self, filename):
        '''
        Restore the models parameters (in-place) from a saved model file

        Args:
            filename (str): saved model path

        Returns:
            `None`: this method only has the side-effect of loading the model parameters from the file
        '''
        return super(Function, self).restore_model(filename)


class UserFunction(Function):
    '''
    Base class of all user extension functions.

    If it has only one output, one can invoke Variable methods on it, which it
    will relay to its only output.

    '''
    def __init__(self, inputs, outputs, op_name, name=''):
        var_inputs = []
        # TODO: this should be done in Swig
        for i in inputs:
            if isinstance(i, cntk_py.Variable):
                var_inputs.append(i)
            elif isinstance(i, cntk_py.Function):
                var_inputs.append(i.output)
            else:
                raise ValueError('expected Variable, but got "%s"'%type(i))

        super(Function, self).__init__(var_inputs, outputs, name, op_name)

    def _forward(self, arguments, outputs, device=None, outputs_to_retain=None):
        '''
        Computes the values of speficied variables in ``outputs``, using values
        provided in ``arguments`` that correspond to each input `Variable` of
        the function whose ``is_input`` is `True`.

        This function calls :func:`forward`, which is to be implemented by the
        user.

        Args:
            arguments (tuple): Value objects of the Function's input
            outputs (iterable): outputs to fetch values for.
            device (:class:`~cntk.device.DeviceDescriptor`, default `None`): the device
             descriptor that contains the type and id of the device on which the
             computation is. If `None`, the default device is used.

        Returns:
             A BackpropState instance, which is used by :func:`backward`.
        '''
        arguments = tuple(value_to_seq(v) for v in arguments)

        map_if_possible(outputs)
        map_if_possible(outputs_to_retain)

        state, results = self.forward(arguments, outputs, device, outputs_to_retain)
        if not isinstance(state, cntk_py.BackPropState):
            state = cntk_py.UserBackPropState(self, device, state)

        for k,v in outputs.items():
            if v is None:
                raise ValueError('not all outputs have been provided')

            # FIXME: seq_starts
            outputs[k] = sanitize_batch(k, v, None, device)

        return state, results

    def _backward(self, state, root_gradients, variables):
        '''
        Backpropagates supplied ``root_gradients`` for one or more of the output
        variables of the Function, to calculate gradients with respect to
        ``variables``. Formally, multiplies the values of ``root_gradients`` by
        the Jacobian of the Function and returns the subset of the output that
        corresponds to ``variables``.

        This function calls :func:`backward`, which is to be implemented by the
        user.

        Example:
            TBD

        Args:
            state (BackPropState): state obtained from a previous call to the
             func:`cntk.ops.Function.forward` method on this Function for the
             computation that this gradient backpropagation corresponds to.
            root_gradients (dict): the gradients that will be backpropagated
            variables (set): a list of input variables with respect to which
             the gradients have to be computed.

        Returns:
            dict: mapping of ``variables`` to NumPy arrays
        '''
        for v in root_gradients:
            root_gradients[v] = value_to_seq(root_gradients[v])
        map_if_possible(variables)

        self.backward(cntk_py.UserBackPropState.data(state), root_gradients, variables)

        for k,v in variables.items():
            if v is None:
                raise ValueError('gradients were not provided for all variables')

            variables[k] = sanitize_batch(k, v, None, state.device())

@typemap
def load_model(filename, device=None):
    '''
    Load the model in ``filename``, that has been saved using
    :func:`~cntk.ops.functions.Function.save_model`.

    Args:
        filename (str): filename to load the model from
        device (:class:`~cntk.device.DeviceDescriptor`, default is the default device):
         instance of DeviceDescriptor

    Returns:
        root node
    '''
    if not device:
        device = DeviceDescriptor.use_default_device()
    return cntk_py.Function.load_model(filename, device)
