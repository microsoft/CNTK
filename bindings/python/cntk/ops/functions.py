from cntk import cntk_py

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
            `None`: text
        '''
        kwargs=dict(locals()); del kwargs['self']; return super(cntk.cntk_py.Function, self).backward(**kwargs)

    @typemap
    def clone(self, parameterCloneMethod='clone', replacements=None)
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
        if replacements is None:
            replacements = dict()
        return super(cntk.cntk_py.Function, self).clone('ParameterCloningMethod_'+parameterCloneMethod.capitalize(), replacements)

    @typemap
    def constants(self):
        '''
        constants
        

        Returns:
            `list`: text
        '''
        return super(cntk.cntk_py.Function, self).constants()

    @typemap
    def eval(self, arguments=None, precision='float', device=None):
        '''
        eval
        

        Returns:
            `None`: text
        '''
        kwargs=dict(locals()); del kwargs['self']; return super(cntk.cntk_py.Function, self).eval(**kwargs)

    @typemap
    def forward(self, arguments, outputs, computeDevice=None, outputsToRetainBackwardStateFor=None)
        '''
        forward
        

        Args:
            arguments (`dict`): text
            outputs (`dict`): text
            computeDevice (`DeviceDescriptor`): text
            outputsToRetainBackwardStateFor (`set`): text
        

        Returns:
            `BackPropStatePtr`: text
        '''
        kwargs=dict(locals()); del kwargs['self']; return super(cntk.cntk_py.Function, self).eval(**kwargs)

    @typemap
    def inputs(self):
        '''
        inputs
        

        Returns:
            `list`: text
        '''
        return super(cntk.cntk_py.Function, self).inputs()

    @typemap
    def name(self):
        '''
        name
        

        Returns:
            `str`: text
        '''
        return super(cntk.cntk_py.Function, self).name()

    @typemap
    def op_name(self):
        '''
        op_name
        

        Returns:
            `str`: text
        '''
        return super(cntk.cntk_py.Function, self).op_name()

    @typemap
    Function.output = lambda self:get_output_and_keep_reference(self)
        '''
        output
        

        Args:
            self.replace_placeholders_internal(ph_map (`ph_map:`): text
        

        Returns:
            `None`: text
        '''
        kwargs=dict(locals()); del kwargs['self']; return super(cntk.cntk_py.Function, self).output(**kwargs)

    @typemap
    def output_internal(self):
        '''
        output_internal
        

        Returns:
            `Variable`: text
        '''
        return super(cntk.cntk_py.Function, self).output_internal()

    @typemap
    def outputs(self):
        '''
        outputs
        

        Returns:
            `list`: text
        '''
        return super(cntk.cntk_py.Function, self).outputs()

    @typemap
    def parameters(self):
        '''
        parameters
        

        Returns:
            `list`: text
        '''
        return super(cntk.cntk_py.Function, self).parameters()

    @typemap
    def placeholders(self):
        '''
        placeholders
        

        Returns:
            `list`: text
        '''
        return super(cntk.cntk_py.Function, self).placeholders()

    @typemap
    def replace_placeholder(self, placeholderReplacement):
        '''
        replace_placeholder
        

        Args:
            placeholderReplacement (`Variable`): text
        

        Returns:
            `FunctionPtr`: text
        '''
        kwargs=dict(locals()); del kwargs['self']; return super(cntk.cntk_py.Function, self).replace_placeholder(**kwargs)

    @typemap
    Function.replace_placeholders = lambda self, ph_map: self.replace_placeholders_internal(ph_map)
        '''
        replace_placeholders
        

        Returns:
            `None`: text
        '''
        kwargs=dict(locals()); del kwargs['self']; return super(cntk.cntk_py.Function, self).replace_placeholders(**kwargs)

    @typemap
    def replace_placeholders_internal(self, placeholderReplacements):
        '''
        replace_placeholders_internal
        

        Args:
            placeholderReplacements (`dict`): text
        

        Returns:
            `FunctionPtr`: text
        '''
        kwargs=dict(locals()); del kwargs['self']; return super(cntk.cntk_py.Function, self).replace_placeholders_internal(**kwargs)

    @typemap
    def restore_from_legacy_model(self, modelFilePath):
        '''
        restore_from_legacy_model
        

        Args:
            modelFilePath (`str`): text
        

        Returns:
            `None`: text
        '''
        kwargs=dict(locals()); del kwargs['self']; return super(cntk.cntk_py.Function, self).restore_from_legacy_model(**kwargs)

    @typemap
    def root_function(self):
        '''
        root_function
        

        Returns:
            `FunctionPtr`: text
        '''
        return super(cntk.cntk_py.Function, self).root_function()