from cntk import cntk_py
from ..utils import typemap, sanitize_var_map, value_to_seq
from enum import Enum, unique

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
            if len(self.outputs) == 1:
                return getattr(self.output, name)

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
               * any other type: if node has an unique input, ``arguments`` is mapped to this input.
                For nodes with more than one input, only dict is allowed.
             In both cases, every every sample in the data will be interpreted
             as a new sequence. To mark samples as continuations of the
             previous sequence, specify ``arguments`` as `tuple`: the
             first element will be used as ``arguments``, and the second one will
             be used as a list of bools, denoting whether a sequence is a new
             one (`True`) or a continuation of the previous one (`False`).
             Data should be either NumPy arrays or a
             :class:`~cntk.io.MinibatchData` instance.
            device (:class:`~cntk.device.DeviceDescriptor`): the device descriptor that
             contains the type and id of the device on which the computation is
             to be performed.

        Returns:
            `bool`: `True` if updates have been performed
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
            arguments: maps variables to their
             input data. The interpretation depends on the input type:

               * dict: keys are input variable or names, and values are the input data.
               * any other type: if node has an unique input, ``arguments`` is mapped to this input.
                For nodes with more than one input, only dict is allowed.
             In both cases, every every sample in the data will be interpreted
             as a new sequence. To mark samples as continuations of the
             previous sequence, specify ``arguments`` as ``tuple``: the
             first element will be used as ``arguments``, and the second one will
             be used as a list of bools, denoting whether a sequence is a new
             one (`True`) or a continuation of the previous one (`False`).
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
            from cntk import DeviceDescriptor
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
        ``variables``.

        Example:
            >>> # compute the value and the derivative of the sigmoid at 0
            >>> v = C.input_variable(shape=(1,))
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
        root_gradients = sanitize_var_map(self.outputs, root_gradients)

        var_gradients = dict((var, None) for var in variables)

        self._backward(state, root_gradients, var_gradients)

        for var, value in var_gradients.items():
            var_gradients[var] = value_to_seq(value)

        return var_gradients

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
    def save_model(self, filename, use_legacy_format=True):
        '''
        Save this function graph into a model file

        Args:
            filename (str): model path
            use_legacy_format (str): if 'True', model is stored using legacy format.
             Otherwise, it's stored using protobuf-based protocol serialization.
        '''
        return super(Function, self).save_model(filename, use_legacy_format)

    @typemap
    def restore_model(self, filename):
        '''
        Restore the models parameters from a saved model file

        Args:
            filename (str): saved model path 

        Returns:
            `None`: this method only has the side-effect of loading the model parameters from the file
        '''
        return super(Function, self).restore_model(filename)

    @property
    @typemap
    def root_function(self):
        '''
        The primitive function at the root of the graph of functions underlying this function.
        '''
        return super(Function, self).root_function()

    @property
    @typemap
    def uid(self):
        '''
        The internally generated unique name of the function.
        '''
        return super(Function, self).uid()

