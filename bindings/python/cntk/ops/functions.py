from cntk import cntk_py
from ..utils import typemap, sanitize_var_map, value_to_seq
from enum import Enum, unique

@unique
class CloneMethod(Enum):
    '''
    Describes different ways how :class:`cntk.ops.functions.Function.forward`
    works.
    '''

    clone = 1
    '''
    New learnable Parameters are created and initialied with the current values of the
    corresponding Parameters of the Function being cloned
    '''

    share = 2
    '''
    Parameters are shared between the Function being cloned and the new clone
    '''

    freeze = 3
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

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]

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
    def clone(self, method=CloneMethod.freeze, substitutions=None):
        '''
        Clones the function. The parameters of the Function are either cloned,
        shared or frozen as specified by the method argument and any variable
        substitutions requested are applied in the cloned Function instance.

        Args:
            method (:class:`cntk.ops.functions.CloneMethod`): one of
             * 'clone': the returned function gets its own copy of parameters (default)
             * 'share': the returned function shares its parameters with this function
             * 'freeze': parameters are cloned and made immutable (constant).
            substitutions (`dict`): a dictionary mapping variables in this
             function to variables in the cloned function

        Returns:
            :class:`Function`: the cloned Function
        '''
        if not isinstance(method, CloneMethod):
            raise ValueError('clone method "%s" is not supported' %
                    str(method))

        method = getattr(cntk_py,
                'ParameterCloningMethod_' + method.name.capitalize())
        if substitutions is None:
            substitutions = {}
        return super(Function, self).clone(method, substitutions)

    @property
    @typemap
    def constants(self):
        '''
        List of all `Constant` variables of this :class:`Function`
        '''
        return super(Function, self).constants()

    def eval(self, arguments=None, device=None):
        '''
        Evaluate the node using the specified `arguments` as input.

        Args:
            arguments (`dict` or `list` or `tuple`): maps variables to their
             input data. The interpretation depends on the input type:
               * `dict`: keys are input variable or names, and values are the input data.
               * `list`: elements are input data in the order their respective variables have been defined in the network.
             In both cases, every every sample in the data will be interpreted
             as a new sequence. To mark samples as continuations of the
             previous sequence, specify `arguments` as `tuple`: the
             first element will be used as `arguments`, and the second one will
             be used as a list of bools, denoting whether a sequence is a new
             one (`True`) or a continuation of the previous one (`False`).
             Data should be either NumPy arrays or a
             :class:`cntk.io.MinibatchData` instance.
            device (:class:`cntk.DeviceDescriptor`): the device descriptor that
             contains the type and id of the device on which the computation is
             to be performed.

        Returns:
            `bool`: `True` if updates have been performed
        '''
        _, output_map = self.forward(arguments or {}, self.outputs, device=device)

        if len(output_map) > 1:
            return output_map
        else:
            return list(output_map.values())[0]

    @typemap
    def forward(self, arguments, outputs, keep_for_backward=None, device=None):
        '''
        Computes the values of speficied variables in `outputs`, using values
        provided in `arguments` that correspond to each input `Variable` of
        the function whose `is_input` is `True`.

        Args:
            arguments (`dict` or `list` or `tuple`): maps variables to their
             input data. The interpretation depends on the input type:
               * `dict`: keys are input variable or names, and values are the input data.
               * `list`: elements are input data in the order their respective variables have been defined in the network.
             In both cases, every every sample in the data will be interpreted
             as a new sequence. To mark samples as continuations of the
             previous sequence, specify `arguments` as `tuple`: the
             first element will be used as `arguments`, and the second one will
             be used as a list of bools, denoting whether a sequence is a new
             one (`True`) or a continuation of the previous one (`False`).
             Data should be either NumPy arrays or a
             :class:`cntk.io.MinibatchData` instance.
            outputs (iterable): outputs to fetch values for.
            keep_for_backward (`set`, default `None): the subset of the
             Function's output variables for which gradients shall be calculated
             in a subsequent backward call. If `None`, the returned state will
             be `None` and a subsequent call to `backward` will not be
             possible.
             for backpropagation.
            device (:class:`cntk.DeviceDescriptor`, default `None): the device
             descriptor that contains the type and id of the device on which the
             computation is. If `None`, the default device is used.

        Returns:
             A tuple (`BackpropState`, `map` of outputs to NumPy arrays). The
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
        Backpropagates supplied `root_gradients` for one or more of the output
        variables of the Function, to calculate gradients with respect to
        `variables`.

        Args:
            state (`BackPropState`): state obtained from a previous call to the
             func:`cntk.ops.Function.forward` method on this Function for the
             computation that this gradient backpropagation corresponds to.
            root_gradients (`dict`): the gradients that will be backpropagated
            variables (`set`): a list of input variables with respect to which
             the gradients have to be computed.

        Returns:
            `dict`: mapping of `variables` to NumPy arrays
        '''
        root_gradients = sanitize_var_map(self.outputs, root_gradients)

        var_gradients = dict((var, None) for var in variables)

        self._backward(state, root_gradients, var_gradients)

        backward_output = {}
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
    @typemap
    def name(self):
        '''
        Name of this function
        '''
        return super(Function, self).name()

    @property
    @typemap
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
    def replace_placeholder(self, substitutions):
        '''
        In-place replace the only placeholder in the function graph with the
        specified substitutions.

        Args:
            substitutions (:class:`cntk.ops.variables.Variable`): the variable that will replace the placeholder

        Returns:
            :class:`Function`: itself

        :raises ExceptionType: when the function has multiple placeholders.
        '''
        return super(Function, self).replace_placeholder(substitutions)

    @typemap
    def restore_from_model(self, filename):
        '''
        Restore the models parameters from a saved model file

        Args:
            filename (`str`): saved model path 

        Returns:
            `None`: this method only has the side-effect of loading the model parameters from the file
        '''
        return super(Function, self).restore_from_legacy_model(filename)

    @property
    @typemap
    def root_function(self):
        '''
        The primitive function at the root of the graph of functions underlying this function.
        '''
        return super(Function, self).root_function()
