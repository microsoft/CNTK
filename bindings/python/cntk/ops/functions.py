from cntk import cntk_py
from ..utils import typemap

class Function(cntk_py.Function):
    '''
    Base class of all operators.

    If it has only one output, one can invoke Variable methods on it, which it
    will relay to its only output.
    '''

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]

        if hasattr(self.output(), name):
            return getattr(self.output(), name)

        raise AttributeError("'%s' object has no attribute '%s'"%\
                (type(self), name))


    @typemap
    def arguments(self):
        '''
        Returns a list of all input variables of the Function that are not of type Parameter or Constant

        Returns:
            `list`: list of input variables
        '''
        return super(cntk.cntk_py.Function, self).arguments()

    @typemap
    def attributes(self):
        '''
        Get the attributes of the function

        Returns:
            `dict`: dictionary of string, value pairs
        '''
        return super(cntk.cntk_py.Function, self).attributes()

    @typemap
    def backward(self, state, rootGradientValues, backPropagatedGradientValuesForInputs):
        '''
        Backpropagates supplied `rootGradientValues` for one or more of the output variables of the Function, to produce gradient Values
        corresponding to the specified set of input variables in `backPropagatedGradientValuesForInputs`. 

        Args:
            state (`BackPropState`): state obtained from a previous call to the Forward method on this Function for the 
              computation that this gradient backpropagation corresponds to
            rootGradientValues (`dict`): the gradients that will be backpropagated to the layer below
            backPropagatedGradientValuesForInputs (`dict`): a dictionary whose keys are `Variable` and whose values one of
              * None: the implementation allocates the actual storage for storing the gradients
              * NDArray: the gradients will be aggregated into this array

        Returns:
            `None`: This method only has side-effects
        '''
        kwargs=dict(locals()); del kwargs['self']; return super(cntk.cntk_py.Function, self).backward(**kwargs)

    @typemap
    def clone(self, parameterCloneMethod='clone', replacements=None):
        '''
        Clones the function. The parameters of the Function are either cloned, shared or frozen as specified by the 
        parameterCloneMethod argument and any variable replacements requested are applied in the cloned Function instance.

        Args:
            parameterCloneMethod (`str`): one of
             * 'clone': the returned function gets its own copy of parameters (default)
             * 'share': the returned function shares its parameters with this function
             * 'freeze': parameters are cloned and made immutable (constant).
            replacements (`dict`): a dictionary mapping variables in this function to variables in the cloned function

        Returns:
            `Function`: the cloned Function
        '''
        if parameterCloneMethod not in set(['clone','share','freeze']):
            raise ValueError('parameterCloneMethod must be one of clone, share, or freeze')
        method = getattr(cntk_py,'ParameterCloningMethod_'+parameterCloneMethod.capitalize())
        if replacements is None:
            replacements = dict()
        return super(cntk.cntk_py.Function, self).clone(method, replacements)

    @typemap
    def constants(self):
        '''
        Returns a list of all `Constant` variables of this `Function`

        Returns:
            `list`: all `Constant` variables of this `Function`
        '''
        return super(cntk.cntk_py.Function, self).constants()

    @typemap
    def eval(self, arguments=None, precision='float', device=None):
        '''
        Evaluate the node using the specified `arguments` as input.

        Args:
            arguments (`dict` or `list` or single input): 
              * map from input variables to the data
              * list of inputs in the order that the function expects or 
              * a single input, if the function only has one argument. 
              Data should be either NumPy arrays or a `:class:cntk.io.MinibatchSource`
            precision (`str` or `np.float32` or `np.float64`): precision, if string
             it can be one of 'float' 'float32, 'double', 'float64', or `None`
            device (:class:`cntk.DeviceDescriptor`): the device descriptor that
             contains the type and id of the device on which the computation is
             to be performed.

        Returns:
            `bool`: `True` if updates have been performed
        '''
        kwargs=dict(locals()); del kwargs['self']; return super(cntk.cntk_py.Function, self).eval(**kwargs)

    @typemap
    def forward(self, arguments, outputs, computeDevice=None, outputsToRetainBackwardStateFor=dict()):
        '''
        Computes and stores the values of speficied variables in `outputs`, using provided `arguments` values corresponding
        to each leaf `Variable` of the function whose is_input() is true.

        Args:
            arguments (`dict`): dictionary of bindings for the input variables
            outputs (`dict`): dictionary of bindings for (a subset of) the output variables. The values in the dictionary can be one of:
              * None: the implementation allocates the actual storage for storing the values
              * NDArray: the values will be written into this array
            computeDevice (`:class:`cntk.DeviceDescriptor`): the device descriptor that
             contains the type and id of the device on which the computation is
             to be performed.
            outputsToRetainBackwardStateFor (`set`): the subset of the Function's output variables for which gradients will be specified
             in a subsequent backward call for backpropagation.

        Returns:
            `BackPropState`: an object containing all intermediate variable values needed during backpropagation of gradients from the 
              `outputsToRetainBackwardStateFor` outputs of the function to any of the inputs of the function, in a subsequent backward call.
        '''
        if computeDevice is None:
            from cntk import DeviceDescriptor
            computeDevice = DeviceDescriptor.use_default_device()

        kwargs=dict(locals()); del kwargs['self']; return super(cntk.cntk_py.Function, self).eval(**kwargs)

    @typemap
    def inputs(self):
        '''
        Returns all input variables of this function.
        

        Returns:
            `list`: all input variables of this function.
        '''
        return super(cntk.cntk_py.Function, self).inputs()

    @typemap
    def name(self):
        '''
        Returns the name of 'this' function.
        

        Returns:
            `str`: the name of 'this' function.
        '''
        return super(cntk.cntk_py.Function, self).name()

    @typemap
    def op_name(self):
        '''
        Returns the name of the operation that this Function denotes
        

        Returns:
            `str`: the name of the operation that this Function denotes
        '''
        return super(cntk.cntk_py.Function, self).op_name()

    # @typemap
    # Function.output = lambda self:get_output_and_keep_reference(self)
        # '''
        # output
        

        # Args:
            # self.replace_placeholders_internal(ph_map (`ph_map:`): text
        

        # Returns:
            # `None`: text
        # '''
        # kwargs=dict(locals()); del kwargs['self']; return super(cntk.cntk_py.Function, self).output(**kwargs)

    # @typemap
    # def output_internal(self):
        # '''
        
        

        # Returns:
            # `Variable`: text
        # '''
        # return super(cntk.cntk_py.Function, self).output_internal()

    @typemap
    def outputs(self):
        '''
        Returns a list consisting of all output variables of this function.
        

        Returns:
            `list`: all output variables of this function
        '''
        return super(cntk.cntk_py.Function, self).outputs()

    @typemap
    def parameters(self):
        '''
        Returns a list of all parameter variables of this function.

        Returns:
            `list`: all parameter variables of this function.
        '''
        return super(cntk.cntk_py.Function, self).parameters()

    @typemap
    def placeholders(self):
        '''
        Returns a list of all placeholders variables of this function.
        

        Returns:
            `list`: all placeholders variables of this function
        '''
        return super(cntk.cntk_py.Function, self).placeholders()

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
        kwargs=dict(locals()); del kwargs['self']; return super(cntk.cntk_py.Function, self).replace_placeholder(**kwargs)

    # @typemap
    # Function.replace_placeholders = lambda self, ph_map: self.replace_placeholders_internal(ph_map)
        # '''
        # replace_placeholders
        

        # Returns:
            # `None`: text
        # '''
        # kwargs=dict(locals()); del kwargs['self']; return super(cntk.cntk_py.Function, self).replace_placeholders(**kwargs)

    # @typemap
    # def replace_placeholders_internal(self, placeholderReplacements):
        # '''
        # replace_placeholders_internal
        

        # Args:
            # placeholderReplacements (`dict`): text
        

        # Returns:
            # `FunctionPtr`: text
        # '''
        # kwargs=dict(locals()); del kwargs['self']; return super(cntk.cntk_py.Function, self).replace_placeholders_internal(**kwargs)

    @typemap
    def restore_from_legacy_model(self, modelFilePath):
        '''
        Restore the models parameters from a saved model file

        Args:
            modelFilePath (`str`): saved model path 

        Returns:
            `None`: this method only has the side-effect of loading the model parameters from the file
        '''
        kwargs=dict(locals()); del kwargs['self']; return super(cntk.cntk_py.Function, self).restore_from_legacy_model(**kwargs)

    @typemap
    def root_function(self):
        '''
        Returns the primitive function at the root of the graph of functions underlying this function.

        Returns:
            `Function`: the primitive function at the root of the graph of functions underlying this function
        '''
        return super(cntk.cntk_py.Function, self).root_function()