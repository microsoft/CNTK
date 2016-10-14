from cntk import cntk_py
from ..utils import typemap, sanitize_var_map, value_to_seq
from enum import Enum, unique

@unique
class CloneMethod(Enum):
    clone = 1
    share = 2
    freeze =3

class Function(cntk_py.Function):
    '''
    Base class of all operators.

    If it has only one output, one can invoke Variable methods on it, which it
    will relay to its only output.
    '''

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]

        if len(self.outputs()) == 1:
            return getattr(self.output(), name)

        raise AttributeError("'%s' object has no attribute '%s'" %
                             (type(self), name))

    @typemap
    def arguments(self):
        '''
        Returns a list of all input variables of the Function that are not of type Parameter or Constant

        Returns:
            `list`: list of input variables
        '''
        return super(Function, self).arguments()

    @typemap
    def attributes(self):
        '''
        Get the attributes of the function

        Returns:
            `dict`: dictionary of string, value pairs
        '''
        return super(Function, self).attributes()

    @typemap
    def clone(self, method=CloneMethod.freeze, replacements=None):
        '''
        Clones the function. The parameters of the Function are either cloned,
        shared or frozen as specified by the method argument and any variable
        replacements requested are applied in the cloned Function instance.

        Args:
            method (:class:`cntk.ops.function.CloneMethod`): one of
             * 'clone': the returned function gets its own copy of parameters (default)
             * 'share': the returned function shares its parameters with this function
             * 'freeze': parameters are cloned and made immutable (constant).
            replacements (`dict`): a dictionary mapping variables in this
             function to variables in the cloned function 

        Returns:
            `Function`: the cloned Function
        '''
        if not isinstance(method, CloneMethod):
            raise ValueError('clone method "%s" is not supported' %
                    str(method))

        method = getattr(cntk_py, 
                'ParameterCloningMethod_' + method.name.capitalize())
        if replacements is None:
            replacements = dict()
        return super(Function, self).clone(method, replacements)

    @typemap
    def constants(self):
        '''
        Returns a list of all `Constant` variables of this `Function`

        Returns:
            `list`: all `Constant` variables of this `Function`
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
        _, output_map = self.forward(arguments or {}, self.outputs(), device=device)

        if len(output_map) > 1:
            return output_map
        else:
            return list(output_map.values())[0]

    @typemap
    def forward(self, arguments, outputs, keep_for_backward=None, device=None):
        '''
        Computes and stores the values of speficied variables in `outputs`,
        using provided `arguments` values corresponding to each leaf `Variable`
        of the function whose is_input() is true. 

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

        in_var_map = sanitize_var_map(self.arguments(), arguments,
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
        root_gradients = sanitize_var_map(self.outputs(), root_gradients)

        var_gradients = dict((var, None) for var in variables)

        self._backward(state, root_gradients, var_gradients)

        backward_output = {}
        for var, value in var_gradients.items():
            var_gradients[var] = value_to_seq(value)

        return var_gradients

    @typemap
    def inputs(self):
        '''
        Returns all input variables of this function.


        Returns:
            `list`: all input variables of this function.
        '''
        return super(Function, self).inputs()

    @typemap
    def name(self):
        '''
        Returns the name of 'this' function.


        Returns:
            `str`: the name of 'this' function.
        '''
        return super(Function, self).name()

    @typemap
    def op_name(self):
        '''
        Returns the name of the operation that this Function denotes


        Returns:
            `str`: the name of the operation that this Function denotes
        '''
        return super(Function, self).op_name()

    # @typemap
    # Function.output = lambda self:get_output_and_keep_reference(self)
        # '''
        # output

        # Args:
        # self.replace_placeholders_internal(ph_map (`ph_map:`): text

        # Returns:
        # `None`: text
        # '''
        # kwargs=dict(locals()); del kwargs['self']; return super(Function,
        # self).output(**kwargs)

    # @typemap
    # def output_internal(self):
        # '''

        # Returns:
        # `Variable`: text
        # '''
        # return super(Function, self).output_internal()

    @typemap
    def outputs(self):
        '''
        Returns a list consisting of all output variables of this function.


        Returns:
            `list`: all output variables of this function
        '''
        return super(Function, self).outputs()

    @typemap
    def parameters(self):
        '''
        Returns a list of all parameter variables of this function.

        Returns:
            `list`: all parameter variables of this function.
        '''
        return super(Function, self).parameters()

    @typemap
    def placeholders(self):
        '''
        Returns a list of all placeholders variables of this function.


        Returns:
            `list`: all placeholders variables of this function
        '''
        return super(Function, self).placeholders()

    @typemap
    def replace_placeholder(self, placeholderReplacement):
        '''
        In-place replace the only placeholder in the function graph with the specified replacement

        Args:
            placeholderReplacement (`Variable`): the variable that will replace the placeholder

        Returns:
            `Function`: itself

        :raises ExceptionType: when the function has multiple placeholders.
        '''
        kwargs = dict(locals())
        del kwargs['self']
        return super(Function, self).replace_placeholder(**kwargs)

    # @typemap
    # Function.replace_placeholders = lambda self, ph_map: self.replace_placeholders_internal(ph_map)
        # '''
        # replace_placeholders

        # Returns:
        # `None`: text
        # '''
        # kwargs=dict(locals()); del kwargs['self']; return super(Function,
        # self).replace_placeholders(**kwargs)

    # @typemap
    # def replace_placeholders_internal(self, placeholderReplacements):
        # '''
        # replace_placeholders_internal

        # Args:
        # placeholderReplacements (`dict`): text

        # Returns:
        # `FunctionPtr`: text
        # '''
        # kwargs=dict(locals()); del kwargs['self']; return super(Function,
        # self).replace_placeholders_internal(**kwargs)

    @typemap
    def restore_from_legacy_model(self, modelFilePath):
        '''
        Restore the models parameters from a saved model file

        Args:
            modelFilePath (`str`): saved model path 

        Returns:
            `None`: this method only has the side-effect of loading the model parameters from the file
        '''
        kwargs = dict(locals())
        del kwargs['self']
        return super(Function, self).restore_from_legacy_model(**kwargs)

    @typemap
    def root_function(self):
        '''
        Returns the primitive function at the root of the graph of functions underlying this function.

        Returns:
            `Function`: the primitive function at the root of the graph of functions underlying this function
        '''
        return super(Function, self).root_function()
