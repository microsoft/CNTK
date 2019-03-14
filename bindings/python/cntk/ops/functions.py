"""
CNTK function constructs. This is the core abstraction of all primitive operators in the CNTK computational graph.
"""


from os import path
from enum import Enum, unique
import sys
import warnings
import collections

import cntk
from cntk import cntk_py, Value
from cntk.device import DeviceDescriptor, cpu
from cntk.internal import map_if_possible, typemap, sanitize_var_map,\
                          sanitize_batch, sanitize_dtype_cntk, _as_tuple,\
                          sanitize_variable_value_dict,\
                          sanitize_Function_attributes,\
                          sanitize_variables_or_functions,\
                          _value_as_sequence_or_array
from cntk.internal.utils import get_python_function_arguments, \
                                map_function_arguments, _py_dict_to_cntk_dict, \
                                _to_cntk_dict_value
from cntk.internal import _UDFDeserializeCallbackWrapper, _serialize
from cntk.internal.sanitize import is_byte_buffer
from ..variables import Record, Variable



@unique
class ModelFormat(Enum):
    '''
    Describes the supported disk format for CNTK model.
    '''

    CNTKv2 = cntk_py.ModelFormat_CNTKv2
    '''
    Default CNTK version 2 format, it supports all CNTK functionalities.
    '''

    ONNX   = cntk_py.ModelFormat_ONNX
    '''
    Open Neural Network Exchange format from https://github.com/onnx/onnx, ONNX currently support
    subset of CNTK functionalities.
    '''

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
    New learnable parameters are created and initialized with the current values of the
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

    `Function` objects can also be constructed directly from a Python lambda,
    by means of the `@Function` decorator.
    The `Function`'s input signature is defined by the lambda.

    Example:

      >>> @Function
      ... def f(x):
      ...     return x * x
      >>> print(f)    # inspect the Function's type
      ElementTimes(x: Sequence[tensor]) -> Sequence[tensor]

    The above form creates a CNTK Function whose arguments are placeholder variables.
    Such a function can only be combined with other symbolic functions.

    To train a Function or pass data to it, you need to declare the types
    of the arguments. In this case, the @Function decorator creates a CNTK Function
    whose arguments are input variables.

    If you use Python 3, Functions with types are declared using Python annotation syntax, e.g.::

      @Function
      def f(x:Tensor[13]):
          return x * x

    If you are working with Python 2.7, use CNTK's :class:`@Signature <cntk.layers.typing.Signature>` decorator instead::

      >>> from cntk.layers.typing import *
      >>> @Function
      ... @Signature(Tensor[13])
      ... def f(x):
      ...     return x * x
      >>> print(f)
      ElementTimes(x: Tensor[13]) -> Tensor[13]

    ``make_block=True`` is an internal parameter used to implement :func:`@BlockFunction <cntk.ops.functions.BlockFunction>`.
    If `BlockFunction()` passes `True`, then the result will be wrapped
    in :func:`~cntk.ops.as_block()`, using the supplied ``op_name`` and ``name`` parameters, which are otherwise ignored.
    '''

    _udf_callback_map = {}
    _deserializer = _UDFDeserializeCallbackWrapper(_udf_callback_map)
    cntk_py._register_udf_deserialize_callback(_deserializer)

    # We override the constructors to implement an overload that constructs
    # a CNTK Functions from a Python function (@Function).
    def __new__(cls, *args, **kwargs):
        if len(args) > 0 and hasattr(args[0], '__call__') and not isinstance(args[0], Function): # overload
            return Function._to_Function(*args, **kwargs)
        return super(Function, cls).__new__(cls) # for some reason, passing *args, **kwargs fails with "object() takes no args

    def __init__(self, *args, **kwargs):
        if len(args) > 0 and hasattr(args[0], '__call__') and not isinstance(args[0], Function): # overload
            return
        super(Function, self).__init__(*args, **kwargs)

    # TODO: bring this back once we have a design for name-accessible .outputs etc.
    #class NamedOutput:
    #    def __init__(self, **kwargs):
    #        for kw in kwargs: # TODO: only allow one arg
    #            self.name = kw
    #            self.arg = kwargs[kw]

    _placeholders_under_construction = set()

    @staticmethod
    def _to_Function_unchecked(f, make_block=False, op_name=None, name=None):
        '''implements @Function decorator; see :class:`~cntk.layers.functions.Function`'''
        f_name = f.__name__ # (only used for debugging and error messages)

        # helper to create a CNTK placeholder or input for a given name
        # An input is created if the parameter is annotated with a Tensor(...) type.
        # In this case, CNTK will immediately trigger type inference.
        # Unannotated parameters will yield placeholder variables instead.
        from .. import placeholder
        def make_arg_variable(name, annotations):
            from ..variables import Variable
            var_type = annotations.get(name, None)
            var_type = Variable._Type._sanitize(var_type)
            if isinstance(var_type, Variable._Type):
                return cntk.input_variable(name=name, **var_type)
            else:
                return placeholder(name=name)

        from ..default_options import default_options
        # Parameter() creation inside code of a Function def is forbidden. Setting 'pure' blocks it in Parameter().
        with default_options(pure=True):

            # get the parameter list through inspection
            arg_names, annotations = get_python_function_arguments(f)

            # The Python function is converted to a CNTK Function by executing it once
            # passing placeholders as inputs. This createss a piece of graph.
            # During execution, the Placeholders of this function are hidden from signatures of any
            # further Functions that may be defined inside this invocation.
            # This is required when @Function definitions are nested, and expression from
            # the outer @Function block is used in an inner block, which would introduce
            # additional Placeholders that will show up as .arguments.
            # This is prevented by (1) maintaining a "invisible placeholders" list,
            # and always filtering .arguments against that list. This is done by the property .signature;
            # i.e. in all of this, do not use .arguments; use .signature instead.
            from .. import combine, alias, as_block
            args = [make_arg_variable(arg_name, annotations) for arg_name in arg_names]

            # helpers
            def force_order_args(fun_args):
                block_args = [placeholder(name=fun_arg.name) for fun_arg in fun_args]   # placeholders inside the BlockFunction
                combined_block_args = combine(block_args)                               # the content of the BlockFunction
                arg_map = list(zip(block_args, fun_args))                               # after wrapping, the block_args map to args
                return as_block(composite=combined_block_args, block_arguments_map=arg_map, block_op_name='Tuple').outputs

            def invoke(fun_args):
                try:
                    # hide Placeholders of this function from .signature() of any function defined inside
                    for arg in args:
                        Function._placeholders_under_construction.add(arg)
                    out = f(*fun_args)
                    if out is None:
                        raise TypeError("CNTK Function '{}' must return a value".format(f_name))
                finally:
                    # unhide Placeholders of this function again
                    for arg in args:
                        Function._placeholders_under_construction.remove(arg)

                # resolve tuples and NamedOutputs  --TODO: check for duplicates
                def resolve_named(output):
                    #if isinstance(output, Function.NamedOutput): # a tuple member is wrapped in a NamedOutput class, we got a name for it
                    #    output = alias(output.arg, name=output.name)
                    # ^^ TODO: Complete the design for name-accessible .outputs, then bring this back.
                    if isinstance(output, cntk_py.Variable):
                        output = combine([output]) # workaround: wrap in another combine() call
                    # TODO: ^^ is this still necessary? Or is this a sanitize() call we need here?
                    return output
                if isinstance(out, tuple): # multi-valued function, returned as a tuple
                    out = [resolve_named(output) for output in out]
                    # BUGBUG: combine() does not allow duplicates, so we wrap them in alias()
                    out_seen = set()
                    for i, out_i in enumerate(out):
                        if out_i in out_seen:
                            out[i] = alias(out_i)
                        else:
                            out_seen.add(out_i)
                    out = combine(out)  # --> turn into a combine()
                else:
                    out = resolve_named(out)
                return out

            # if called from BlockFunction() then wrap into a block
            if make_block: # if we make a block then run off a separate set
                block_args = [make_arg_variable(arg.name, annotations) for arg in args]  # placeholders inside the BlockFunction
                out = invoke(block_args)
                out = as_block(composite=out, block_arguments_map=list(zip(block_args, args)), block_op_name=op_name, block_instance_name=name)
            # not a block: ensure parameter ordering
            else:
                fun_args = args
                #if len(fun_args) > 1:
                #    fun_args = force_order_args(fun_args)
                # BUGBUG: Python interpreter crashes sometimes with this enabled, so for now fix it after the fact only if needed
                # now invoke the Python function
                out = invoke(fun_args)
                # BUGBUG workaround: fix it after the fact with an inefficient solution only if we got it wrong
                out_arg_names = [arg.name for arg in out.signature]
                if set(out_arg_names) == set(arg_names) and out_arg_names != arg_names:  # order came out wrong
                    fun_args = force_order_args(fun_args)
                    out = invoke(fun_args)

            return out, args

    @staticmethod
    def _sanitize_check_Function(f_out, f_args, f):
        arg_names, annotations = get_python_function_arguments(f)
        #verify the argument length first
        if len(f_out.signature) != len(f_args):
            f_name = f.__name__
            unfulfilled_args = set(f_out.signature) - set(f_args)
            if unfulfilled_args:
                unfulfilled_arg_names = [arg.name for arg in unfulfilled_args]
                raise TypeError(
                    "CNTK Function '{}' has {} missing arguments ({}), which is currently not supported".format(f_name,
                                                                                                                len(
                                                                                                                    unfulfilled_arg_names),
                                                                                                                ", ".join(
                                                                                                                    unfulfilled_arg_names)))
            else:
                unused_args = set(f_args) - set(f_out.signature)
                unused_arg_names = [arg.name for arg in unused_args]
                raise TypeError(
                    "CNTK Function '{}' has {} unused arguments ({}), which is currently not supported".format(f_name,
                                                                                                               len(
                                                                                                                   unused_arg_names),
                                                                                                               ", ".join(
                                                                                                                   unused_arg_names)))

        #then verify that we got the parameter order right
        out_arg_names = [arg.name for arg in f_out.signature]
        assert out_arg_names == arg_names, (out_arg_names, arg_names)
        return f_out

    @staticmethod
    def _to_Function(f, make_block=False, op_name=None, name=None):
        out, args = Function._to_Function_unchecked(f, make_block, op_name, name)
        return Function._sanitize_check_Function(out, args, f)

    @property
    def signature(self):
        '''
        Returns the signature of a Function.
        This is the .arguments[] list without placeholders that belong to an outer, not yet completed @Function def.
        '''
        sig = [arg for arg in self.arguments if arg not in Function._placeholders_under_construction]
        return tuple(sig)

    def argument_map(self, *args, **kwargs):
        '''
        Determines the {placeholder: variable} map for use with various call operations
        Returns a dictionary from this function's placeholders to whatever arguments are passed.
        Accepted are both positional and keyword arguments.
        This mimics Python's argument interpretation, except that keyword arguments are not optional
        (there is no concept of default value).
        This does not require the arguments to be Variables or Functions. It is also called by train_minibatch().
        '''
        params = self.signature    # function parameters
        if len(args) + len(kwargs) != len(params):
            raise TypeError("CNTK Function expected {} arguments, got {}".format(len(params), len(args) + len(kwargs)))
        params_dict = { arg.name: arg for arg in params }
        return map_function_arguments(params, params_dict, *args, **kwargs)

    @staticmethod
    def _replace_args_type_check(arg_map): # type: (Dict[param: Variable, arg: Variable]), param meant to be substituted by arg
        '''
        Performs a type-compatibility check for arguments to replace_placeholders() and clone(),
        in order to output an actionable error message in case of an error.
        '''
        for i, arg_map_item in enumerate(arg_map.items()):
            param = arg_map_item[0]  # parameter = what gets substituted
            arg   = arg_map_item[1]  # argument  = what it gets substituted with
            #print('checking param', param.name, 'against arg', arg.name)
            param_type = param._type
            arg_type   = arg._type if isinstance(arg, cntk_py.Variable) else arg.output._type if isinstance(arg, Function) else None
            def param_name(): # helper to get a descriptive name for param
                if param.name:
                    return "argument %s" % param.name
                else:
                    return 'positional argument %d' % i
            if not arg_type:
                raise TypeError(param_name() + " was passed an object that is not a Variable or Function")
            # parameter shape is not yet known, any input is acceptable
            if not param_type.shape_is_known or param.is_placeholder:
                # Note: if a Function with nown inputs gets cloned while replacing the inputs
                # with placeholders, those placeholders retain their shapes for some reason.
                # But in this case, it should be allowed to replace them with mismatching dimensions,
                # hence we do not test placeholders, only inputs.
                # TODO: Should clone-replacing inputs with placeholders reset the shapes to unknown?
                continue
            if not arg_type.shape_is_known:
                raise TypeError(param_name() + ' has a known shape, and cannot be passed a Variable of unknown shape')
            # TODO: add tests for this complex condition
            if len(arg_type.shape) < len(param_type.shape) or \
                   arg_type.shape[-len(param_type.shape):] != param_type.shape or \
                   (arg_type.dynamic_axes and arg_type.dynamic_axes != param_type.dynamic_axes) or \
                   arg_type.dtype != param_type.dtype or \
                   arg_type.is_sparse != param_type.is_sparse:
                raise TypeError(param_name() + "'s type " + str(param_type) + " is incompatible with the type " + str(arg_type) + " of the passed Variable")

    def update_signature(self, *arg_types, **kwarg_types):
        '''
        Defines input shapes, in-place
        e.g.
        model.update_signature(42)
        pass a list of objects that define the dimensions etc. of the placeholders
        Currently you can pass an int, a tuple, an Input, or a dict created with Type()
        '''
        arg_map = self.argument_map(*arg_types, **kwarg_types) # map type specs to Function parameters
        def to_input(arg_type, name):
            #from cntk import input
            from ..variables import Variable
            if isinstance(arg_type, (int, tuple)): # just passed a shape
                return cntk.input_variable(shape=_as_tuple(arg_type), name=name)
            arg_type = Variable._Type._sanitize(arg_type)
            if isinstance(arg_type, Variable._Type): # full type given as Tensor[...] etc.
                return cntk.input_variable(name=name, **arg_type)
            raise TypeError("update_signature() expects arguments of type int, tuple of int, or Type.Variable")
        # map the given types:
        #  - create an Input with the given Type or shape
        #  - keep the name property of the Function parameter
        #  - skip argument types passed as None
        arg_map = { param: to_input(arg_type, name=param.name) for param, arg_type in arg_map.items() if arg_type is not None }
        Function._replace_args_type_check(arg_map)
        self.replace_placeholders(arg_map)


    def declare_args(self, *arg_types):
        '''
        Back-compat wrapper for update_signature() (beta12 and before).
        '''
        warnings.warn('This will be removed in future versions. Please use '
                'update_signature(...) instead', DeprecationWarning)
        placeholders = self.placeholders  # the unbound parameters to fill in
        if len(arg_types) != len(placeholders):
            raise TypeError("CNTK Function.declare_args() expected {} arguments, got {}".format(len(placeholders), len(arg_types)))
        def to_input(arg):
            if isinstance(arg, cntk_py.Variable):
                return arg
            else:
                #from cntk import input
                return cntk.input_variable(arg)

        args = [to_input(arg) for arg in arg_types]
        arg_map = dict(zip(placeholders, args))
        Function._replace_args_type_check(arg_map)
        self.replace_placeholders(arg_map)


    def __call__(self, *args, **kwargs):
        '''
        Call a Function, either on symbolic or numeric inputs.

           * If at least one input is a CNTK Function or Variable, then
             result is a CNTK Function object, with inputs bound to the arguments.
             This is a short-hand for `f.clone(share, argument_map(*args, **kwargs))`.
           * Otherwise, all arguments must be numbers, numpy arrays, or a :class:`~cntk.io.MinibatchData` instance.
             Then perform the actual computation and return the numeric result.
             This is a short-hand for `f.eval(argument_map(*args, **kwargs))`,
             except that there is no `device` parameter. If you need that, use `eval()` directly.

        Args:
            *args, **kwargs: The arguments to pass to the Function.

        Returns:
             In case of symbolic inputs, returns another CNTK Function object with inputs bound to the arguments.
             Otherwise returns a tuple of numpy arrays for tuple-valued Functions, and a single numpy array otherwise.
        '''

        # parse argument list and map to the function's input
        arg_map = self.argument_map(*args, **kwargs)

        # if placeholders were excluded due to being under construction,
        # we must include them in the argmap, otherwise they will be cloned
        for arg in self.arguments:
            if arg not in arg_map:
                arg_map[arg] = arg

        # determine whether this is eval() or clone()
        is_symbolic = any(isinstance(arg, (cntk_py.Function, cntk_py.Variable)) for arg in arg_map.values())

        # symbolic: return a cloned Function
        # applying the function means to inline its piece of graph
        if is_symbolic:
            Function._replace_args_type_check(arg_map)
            return self.clone(CloneMethod.share, arg_map)

        # numeric: evaluate
        outputs = self.outputs
        _, output_map = self.forward(arg_map, outputs)
        assert len(output_map) == len(outputs), (output_map, outputs)
        if len(output_map) > 1: # tuple-valued: return tuple
            return tuple(output_map[output] for output in outputs)
        else: # single value: return numpy array and that's it
            return list(output_map.values())[0]

    # TODO: remove the parallel application; instead
    #  - function tuples always operate on all inputs, just as if they were a single function
    #  - parallel application would be done by nested Sequential or >> expressions
    #  - we also need to rethink Sequential() for the case that the first function passed to
    #    it accepts multiple arguments. That should just become the returned composite's signature.
    #    It naturally would if we just passed it on to Function, but in case of a tuple, we'd need
    #    to create intermediate placeholders so that all functions in the tuple get to share the inputs.
    def __rshift__(self, other):
        '''
        Forward function composition (G o F), same as Sequential([F, G]).
        Unlike __call__(), __rshift__() accepts tuples:

         * `G` can be a tuple of Functions. They are applied in parallel, yielding a tuple result.
           If `F` is a single-valued Function, it will be fed to all items.
         * if `F` is a tuple-valued Function piped and `G` is a single Function, the tuple
           values will be used as the arguments to `G`.
         * if both are tuples, they are applied 1:1

        E.g. `Embedding(500) >> (Recurrence(500), Recurrence(500, go_backwards=True)) >> splice >> Dense`
        '''
        inputs = self.outputs
        input_is_tuple = len(inputs) > 1
        # if piping into a tuple of Functions, apply item-wise
        if isinstance(other, tuple):
            from cntk import combine
            return combine([other[i](inputs[i if input_is_tuple else 0]) for i in range(len(other))])
        # if applying a single function to a tuple-valued Function, pass the items as the args
        elif input_is_tuple:
            return other(*inputs)
        # regular case: one input, one Function
        else:
            return other(self)

    def __lshift__(self, other):
        '''
        Backward function composition (self o other)
        '''
        return self(other)

    def __getattr__(self, name):
        '''
        Access a member inside this object.
        Members of ``Function`` can be accessed directly.
        In addition, members of the Function's output, if only one, are accessed here.
        Lastly, this also gives access to Functions and Variables inside this Function's
        graph by their user-specified name, e.g. ``model.embed.E``, as long as those names are not also
        member names of Function or Variable.
        '''
        # If name is not a member of Function or Variable, first look for
        # a user-named item in the graph.
        # (Known member names cannot be overridden by user-named items,
        # to ensure that the API functions.)
        if not hasattr(Variable, name) and not hasattr(Function, name) \
           and not name.startswith('_') and name not in ['outputs', 'output', 'this']:
            # lookup of a named object inside the graph
            # When 'self' is a BlockFunction (e.g. a named layer), then we only search in there,
            # while when 'self' is a regular node (e.g. a named output using Label),
            # we search the composite, which may return multiple hits with the same name.
            # In case of multiple matches, we fail.
            # BUGBUG: That is a problem if, e.g., someone used a layer (=BlockFunction) twice
            # and then looks it up by name, as that will fail although both instances are identical.
            from cntk.logging.graph import find_by_name
            root = self.block_root if self.is_block else self
            item = typemap(find_by_name)(root, name, depth=1)
            if item:
                return item

        # If something is not found in Function, look it up in its output
        # variable, if it has only one.
        if name.startswith('_') or name in ['outputs', 'output', 'this']:
            # These should not be looked up in self's output.
            # 'outputs' and 'output' are required to fetch the attribute for
            # in the Variable.
            # 'this' is required for Swig and needs to be thrown if the
            # object is created the first time.
            raise AttributeError("neither Function nor its output variable"
                    " has '%s'"%name)

        # access an API member of 'output', such as .shape()
        outputs = self.__getattribute__('outputs')
        if len(outputs) != 1:
            raise AttributeError("Function does not have '%s' and it cannot "
                    "be looked up in its outputs because it does not have "
                    "exactly one"%name)

        return getattr(outputs[0], name)

    @property
    def type(self):
        '''
        Get type of a Function's output.
        '''
        return self.output.type

    @property
    @typemap
    def arguments(self):
        '''
        List of all input variables of the Function that are not of type Parameter or Constant.
        
        Note that due to the different matrix storage format in C++(column major) and Python(row major),
        the order of arguments for some ops(Times, TransposeTimes, and Gemm) in C++ and Python are not the same. 
        In previous CNTK versions, the default for this api was to return arguments in C++ order. 
        Now the default for this api is set to python order. This way it will return arguments in the same order as they are fed into ops.
        If you wish to still get arguments in C++ order, you can simply override the global option.
        
        Example:
         >>> import cntk as C
         >>> a = C.input_variable((3,4), name='a')
         >>> b = C.input_variable((4,5), name='b')
         >>> c = C.times(a, b)
         >>> c.arguments    # python order
             (Input('a', [#], [3 x 4]), Input('b', [#], [4 x 5]))

         >>> from cntk.default_options import set_global_option
         >>> set_global_option('python_operand_order', False)
         >>> c.arguments    # C++ order
             (Input('b', [#], [4 x 5]), Input('a', [#], [3 x 4]))

        '''
        from ..default_options import get_global_option
        python_operand_order = get_global_option('python_operand_order', True)
        return super(Function, self).arguments(python_operand_order)

    @property
    @typemap
    def attributes(self):
        '''
        List of the attributes of the function
        '''
        return sanitize_Function_attributes(super(Function, self).attributes())

    def set_attribute(self, name, value):
        '''
        Allows to change a function attribute.

        Args:
            name (string): one of

             * 'dropoutRate': modifies the dropout rate of a dropout function
               (can only be invoked on a function instance returned either from
               :func:`~cntk.ops.dropout` or :func:`find_by_name`).

             * 'rngSeed': modifies the seed of a stateful function (can only be
               invoked on  function instance returned from :func:`~cntk.ops.dropout`,
               :func:`~cntk.ops.random_sample`,
               :func:`~cntk.ops.random_sample_inclusion_frequency` or :func:`find_by_name`)

            value (float in case of 'dropoutRate', int for 'rngSeed'): the new value
             of the corresponding attribute.
        '''
        value = _to_cntk_dict_value(value)
        return super(Function, self).set_attribute(name, value)

    def _get_or_reset_custom_attributes(self, reset):
        '''
        Internal non-property version of custom attribute
        Note that composite function does not have custom attributes, so the property returns its root_function's custom_attributes.

        Args:
            reset (bool): whether to reset the dictionary
        '''
        if self.is_composite:
            return self.root_function._get_or_reset_custom_attributes(reset)
        else:
            if reset:
                super(Function, self).reset_custom_attributes()
            return super(Function, self).get_custom_attributes()

    @property
    def custom_attributes(self):
        '''
        Get function custom attributes in cntk_py.Dictionary for both read and write.
        '''
        return self._get_or_reset_custom_attributes(reset=False)

    @custom_attributes.setter
    def custom_attributes(self, values):
        '''
        Set function custom attributes in a batch, and drops old attributes

        Args:
            values (dict): a dictionary of new custom attributes
        '''
        values = values or {}
        if not isinstance(values, dict):
            raise TypeError("values must be a dictionary")

        custom_attr = self._get_or_reset_custom_attributes(reset=True)
        for key in values.keys():
            custom_attr[key] = values[key]

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
        # C++ clone() can only clone composites. If we are not a composite, make it one using combine()
        if not self.is_composite:
            from cntk import combine
            return combine([self]).clone(method, substitutions)

        method = getattr(cntk_py,
                'ParameterCloningMethod_' + CloneMethod(method).name.capitalize())
        substitutions = substitutions or {}
        if not isinstance(substitutions, dict):
            raise TypeError("Variable substitution map must be a dictionary")
        for prev_node, new_node in substitutions.items():
            if not new_node or not prev_node:
                raise AttributeError("Cannot replace node: " + str(prev_node) + " with node: " + str(new_node) + ". Neither node can be None.")
        return super(Function, self).clone(method, substitutions)

    @property
    @typemap
    def constants(self):
        '''
        List of all `Constant` variables of this :class:`~cntk.ops.functions.Function`
        '''
        return super(Function, self).constants()

    def eval(self, arguments=None, outputs=None, device=None, as_numpy=True):
        '''
        Evaluate the Function's outputs using the specified ``arguments`` as input.

        Args:
            arguments: maps variables to their input data. The interpretation depends on
             the input type:

              * dict: keys are input variable or names, and values are the input data.
                See :meth:`~cntk.ops.functions.Function.forward` for details on passing
                input data.
              * any other type: if node has a unique input, arguments is
                mapped to this input.

             For nodes with more than one input, only dict is allowed.

             In both cases, every sample in the data will be interpreted
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
            outputs (iterable, optional): outputs to fetch values for. If not
             set, all outputs of the function will be fetched.
            device (:class:`~cntk.device.DeviceDescriptor`): the device descriptor that
             contains the type and id of the device on which the computation is
             to be performed.
            as_numpy (bool): whether to return the result as a NumPy array. Default True.
             Specifying this as False returns a CNTK Value which avoids a
             costly conversion but returns a somewhat opaque object. Also, the Value objects
             are temporary and only guaranteed to be valid until the next forward/eval/backward/grad call.
             You must explicitly clone the temporay Value objects if they need to be accessed later.

        Note:
             See :meth:`~cntk.ops.functions.Function.forward` for examples on
             passing input data.

        Returns:
           dict or NumPy Array: Dict with keys of output variable names and values of
           output variable. A single NumPy array if there is only one output value.
        '''
        if outputs is None:
            outputs = self.outputs

        _, output_map = self.forward(arguments, outputs, device=device, as_numpy=as_numpy)
        return sanitize_variable_value_dict(output_map)

    @typemap
    def forward(self, arguments, outputs=None, keep_for_backward=None, device=None, as_numpy=True):
        '''
        Computes the values of speficied variables in ``outputs``, using values
        provided in ``arguments`` that correspond to each input `Variable` of
        the function (i.e. those that have ``is_input = True``).

        Example:
            >>> # Example of passing dense data
            >>> v = C.input_variable(shape=(3,))
            >>> f = C.reciprocal(v)
            >>> _, fv = f.forward({v:[[1, 2, 4]]})
            >>> list(fv.values())[0]
            array([[ 1.  ,  0.5 ,  0.25]], dtype=float32)

        Example:
            >>> # Passing sparse values as one-hot with a vocabulary size of 5
            >>> vocab_size = 5
            >>> v = C.sequence.input_variable(shape=(vocab_size,), is_sparse=True)
            >>> f = C.times(v, np.eye(vocab_size))
            >>> # Passing a batch of two sequences:
            >>> # 1st sequence: word 1
            >>> # 2nd sequence: words 2 and 4
            >>> batch = [[1],[2,4]]
            >>> sparse_batch = C.Value.one_hot(batch, vocab_size)
            >>> _, fv = f.forward({v:sparse_batch})
            >>> list(fv.values())[0]
            [array([[ 0.,  1.,  0.,  0.,  0.]], dtype=float32),
             array([[ 0.,  0.,  1.,  0.,  0.], [ 0.,  0.,  0.,  0.,  1.]], dtype=float32)]

        Example:
            >>> # Doing the same, but with a CSR matrix from scipy.sparse
            >>> vocab_size = 5
            >>> from scipy.sparse import csr_matrix
            >>> v = C.sequence.input_variable(shape=(vocab_size,), is_sparse=True)
            >>> f = C.times(v, np.eye(vocab_size))
            >>> # Note that csr_matrix automatically uses a sparse representation underneath.
            >>> sparse_batch = [csr_matrix([[0,1,0,0,0]]), csr_matrix([[0,0,1,0,0], [0,0,0,0,1]])]
            >>> _, fv = f.forward({v:sparse_batch})
            >>> list(fv.values())[0]
            [array([[ 0.,  1.,  0.,  0.,  0.]], dtype=float32),
             array([[ 0.,  0.,  1.,  0.,  0.], [ 0.,  0.,  0.,  0.,  1.]], dtype=float32)]
            <BLANKLINE>
            >>> # Much more efficient, however, is to incrementally create CSR arrays.
            >>> # See https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
            >>> # for more information.
            >>> def seq_to_csr_matrix(seq, vocab_size):
            ...     indptr = [0]
            ...     indices = []
            ...     data = []
            ...     for term_idx in seq:
            ...         indices.append(term_idx)
            ...         data.append(1)
            ...         indptr.append(len(indices))
            ...     return csr_matrix((data, indices, indptr), shape=(len(seq), vocab_size))
            >>> sparse_batch = [seq_to_csr_matrix(seq, vocab_size) for seq in batch]
            >>> _, fv = f.forward({v:sparse_batch})
            >>> list(fv.values())[0]
            [array([[ 0.,  1.,  0.,  0.,  0.]], dtype=float32),
             array([[ 0.,  0.,  1.,  0.,  0.], [ 0.,  0.,  0.,  0.,  1.]], dtype=float32)]


        Args:
            arguments: maps variables to their input data. The interpretation depends on
             the input type:

              * dict: keys are input variable or names, and values are the
                input data. To specify a minibatch, provide a list of arrays.
                The shape of each array must be compatible with the shape of
                the dictionary key. If the array denotes a sequence then the
                elements of the sequence are grouped along axis 0.
              * any other type: if node has a unique input, arguments is
                mapped to this input.

             For nodes with more than one input, only dict is allowed.

             In both cases, every sample in the data will be interpreted
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
            outputs (iterable, optional): outputs to fetch values for. If not
             set, all outputs of the function will be fetched.
            keep_for_backward (set, default `None`): the subset of the
             Function's output variables for which gradients shall be calculated
             in a subsequent backward call. If `None`, the returned state will
             be `None` and a subsequent call to :func:`backward` will not be
             possible.
            device (:class:`~cntk.device.DeviceDescriptor`, default `None`): the device
             descriptor that contains the type and id of the device on which the
             computation is. If `None`, the default device is used.
            as_numpy (bool): whether to return the result as a NumPy array. Default True.
             Specifying this as False returns a CNTK Value which avoids a
             costly conversion but returns a somewhat opaque object. Also, the Value objects
             are temporary and only guaranteed to be valid until the next forward/eval/backward/grad call.
             You must explicitly clone the temporay Value objects if they need to be accessed later.

        Returns:
             A tuple (BackPropState, map of outputs to NumPy arrays). The
             BackPropState is a handle taken by :func:`backward`.
        '''
        if device is None:
            device = DeviceDescriptor.use_default_device()

        in_var_map = sanitize_var_map(self.arguments, arguments,
                                      None, device)
        if outputs is None:
            outputs = self.outputs
        else:
            outputs = sanitize_variables_or_functions(outputs)

        output_map = {v: None for v in outputs}
        keep_for_backward = set(keep_for_backward or {})

        state = super(Function, self)._forward(in_var_map, output_map, device,
                                               keep_for_backward)
        if as_numpy:
            for k, v in output_map.items():
                output_map[k] = _value_as_sequence_or_array(v, k)

        return state, output_map

    @typemap
    def backward(self, state, root_gradients, variables, as_numpy=True):
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
            array([[ 0.5]], dtype=float32)
            >>> list(grad.values())[0]
            array([[ 0.25]], dtype=float32)

        Args:
            state (BackPropState): state obtained from a previous call to the
             func:`cntk.ops.Function.forward` method on this Function for the
             computation that this gradient backpropagation corresponds to.
            root_gradients (dict): the gradients that will be backpropagated
            variables (set): a list of input variables with respect to which
             the gradients have to be computed.
            as_numpy (bool): whether to return the gradients as a NumPy array. Default True.
             Specifying this as False returns a CNTK Value which avoids a
             costly conversion but returns a somewhat opaque object. Also, the Value objects
             are temporary and only guaranteed to be valid until the next forward/eval/backward/grad call.
             You must explicitly clone the temporay Value objects if they need to be accessed later.

        Note:
             See :meth:`~cntk.ops.functions.Function.forward` for more examples
             on passing input data.

        Returns:
            dict: mapping of ``variables`` to NumPy arrays
        '''
        if state is None:
            raise ValueError('You are attempting to backpropagate on a '
                'minibatch for which the corresponding forward operation did not '
                'keep any intermediate results, Please set keep_for_backward in '
                'forward to the variables in root_gradients.keys()')
        device = state.device()
        root_gradients = sanitize_var_map(self.outputs, root_gradients,
                                          None, device)

        var_gradients = {var: None for var in variables}

        self._backward(state, root_gradients, var_gradients)

        if as_numpy:
            for var, value in var_gradients.items():
                var_gradients[var] = _value_as_sequence_or_array(value, var)

        return var_gradients

    @typemap
    def grad(self, at, wrt=None, outputs=None, device=None, as_numpy=True, grad_root=None):
        '''
        Computes the gradient of this Function at location ``at`` with respect to ``wrt``.
        The Function must have a single output.

        Example:
            >>> x = C.input_variable(shape=(1,), needs_gradient=True)
            >>> y = C.sqrt(x)
            >>> a = np.asarray([1,4,16],dtype=np.float32).reshape(3,1)
            >>> y.grad({x:a})
            array([[ 0.5  ],
            <BLANKLINE>
                   [ 0.25 ],
            <BLANKLINE>
                   [ 0.125]], dtype=float32)

        Args:
            at (dict) : mapping of the Function's arguments to values
            wrt (list, default `None`): list of Variables with respect to which the
             gradient will be computed. If omitted, the gradients with
             respect to all arguments of this Function that need gradient will be computed.
            outputs (iterable, optional): outputs (including intermediate outputs in the graph)
             to fetch values for. If not specified, values for none of the outputs are fetched.
            device (:class:`~cntk.device.DeviceDescriptor`, default `None`): the device
             descriptor that contains the type and id of the device on which the
             computation is performed. If `None`, the default device is used.
            as_numpy (bool, default `True`): whether to return the gradients as a NumPy array. Default True.
             Specifying this as False returns a CNTK Value which avoids a
             costly conversion but returns a somewhat opaque object. Also, the Value objects
             are temporary and only guaranteed to be valid until the next forward/eval/backward/grad call.
             You must explicitly clone the temporay Value objects if they need to be accessed later.
            grad_root (:class:`~cntk.variables.Variable`, optional): specify the root of gradients calculation.
             If not specified, the output of this function will be used as gradient root.

        Returns:
            dict or NumPy Array or a tuple of these: Dict with keys of ``wrt`` variables and gradient values of
            ``wrt`` variables. A single NumPy array if there is only one gradient value.
            If ``outputs`` were specified (to fetch values for), this method returns a tuple where the 2nd element
            of the tuple is the ``outputs`` values; a dict with keys of specified ``outputs`` variables and
            values of computed ``outputs``, or a single NumPy array if there is only one output value.
            Each element has the same shape as the ``wrt`` or ``outputs`` variables including dynamic axes
            (such as the batch axis).
        '''
        if device is None:
            device = DeviceDescriptor.use_default_device()

        in_var_map = sanitize_var_map(self.arguments, at, None, device)

        if outputs is None:
            outputs = []

        if wrt is None:
            wrt = [arg for arg in self.arguments if arg.needs_gradient]
            if len(wrt) == 0:
                raise ValueError("None of the Function '%s' arguments have 'needs_gradient == True'" % str(self))

        output_map = {v: None for v in outputs}
        wrt_map = {v: None for v in wrt}

        if grad_root is None:
            super(Function, self).gradients(in_var_map, wrt_map, output_map, device)
        else:
            super(Function, self).gradients(in_var_map, grad_root, wrt_map, output_map, device)

        if as_numpy:
            for k in output_map:
                output_map[k] = _value_as_sequence_or_array(output_map[k], k)
            for k in wrt_map:
                wrt_map[k] = _value_as_sequence_or_array(wrt_map[k], k)

        if len(output_map) == 0:
            return sanitize_variable_value_dict(wrt_map)
        else:
            return sanitize_variable_value_dict(wrt_map), sanitize_variable_value_dict(output_map)

    @property
    @typemap
    def inputs(self):
        '''
        List of variables that are inputs of this function.
        Note that 'inputs' here denotes all Variables that feed into this Function
        including any Parameter/Constant Variables that are children of this Function.
        '''
        return super(Function, self).inputs(True)

    @property
    def name(self):
        '''
        Name of this function

        Args:
          getter (str): returns the name of the function.
          setter (str): sets the name of the function. Setting the name of a
           Function is only allowed if the Function does not already have a
           name. Calling this method, when this Function already has a name,
           results in an exception.
        '''
        return super(Function, self).name()

    @name.setter
    def name(self, function_name):
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
        Throws an exception if this is not a block Function.
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

    def print_node_timing(self):
        '''
        Prints per-node average timing per-minibatch for each primitive function.
        statistics would reset after print
        '''
        return super(Function, self).print_node_timing()



    def __str__(self):
        '''
        Describes the Function and its signature as a string.

        Example:
         >>> f = C.log(C.input(1), name='f') # Function constructed as a graph
         >>> print(f)
         f: Log(Tensor[1]) -> Tensor[1]
         >>> d = C.layers.Dense(10) # Function constructed as a layer
         >>> print(d)
         Dense(x: Sequence[tensor]) -> Sequence[tensor]
         >>> @C.Function   # construct a primitive Function through @Function
         ... def g(x,y):
         ...     return x+y
         >>> print(g)
         Plus(x: Sequence[tensor], y: Sequence[tensor]) -> Sequence[tensor]
         >>> @C.Function   # construct a composite through @Function
         ... def h(x,y):
         ...     return C.exp(x+y)
         >>> print(h)
         Composite(x: Sequence[tensor], y: Sequence[tensor]) -> Sequence[tensor]
        '''
        f_name = self.name
        op_name = self.op_name
        if self.is_composite:
            if self.root_function and all(i.uid == ri.uid for i, ri in zip(self.inputs, self.root_function.inputs)):
                op_name = self.root_function.op_name
            else:
                op_name = 'Composite' # (real op_name is CompositeFunctionOpName)
        else:
            op_name = self.op_name

        args = self.signature
        def format_arg_spec(v, is_output=False):
            s = v.name + ': ' if not is_output and v.name else ''  # (suppress output names, since they duplicate the function name)
            return s + str(v._type)
        outputs = self.outputs
        if len(outputs) > 1:
            output_signature = 'Tuple[' + ', '.join(format_arg_spec(output, True) for output in outputs) + ']'
        else:
            output_signature = format_arg_spec(outputs[0], True)
        if self.name:
            f_name += ": "
        return f_name + op_name + '(' + ", ".join([format_arg_spec(param) for param in args]) + ') -> ' + output_signature



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
        substitutions = substitutions or {}
        if not isinstance(substitutions, dict):
            raise TypeError("Variable substitution map must be a dictionary")
        return super(Function, self).replace_placeholders(substitutions)

    @typemap
    def replace_placeholder(self, substitution):
        '''
        In-place replace the only placeholder in the function graph with the
        specified substitution.

        Args:
            substitution (:class:`~cntk.variables.Variable`): the variable
             that will replace the placeholder

        Returns:
            :class:`Function`: itself

        :raises Exception: when the function has multiple placeholders.
        '''
        return super(Function, self).replace_placeholder(substitution)

    @typemap
    def find_all_with_name(self, name, depth=0):
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
            depth (int, default 0): how deep into the block hierarchy the DFS
             algorithm should go into. Set to -1 for infinite depth.

        Returns:
            list of :class:`Function` objects matching ``name``

        See also:
            :func:`find_by_name`
        '''
        from cntk.logging import graph
        return graph.find_all_with_name(self, name, depth)

    # TODO have a better name for combine() in this case
    @typemap
    def find_by_name(self, name, depth=0):
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
            array([[ 3.]], dtype=float32)

        Args:
            name (str): names to look for
            depth (int, default 0): how deep into the block hierarchy the DFS
             algorithm should go into. Set to -1 for infinite depth.

        Returns:
            :class:`Function` object matching ``name``

        See also:
            :func:`find_all_with_name`
        '''
        from cntk.logging import graph
        return graph.find_by_name(self, name, depth)

    class _ProgressCollector(cntk_py.ProgressWriter):
        '''
        Internal helper for tracking loss and metric values for train() and test().
        '''
        # TODO: If this is of general interest, consider to move it to progress_print.py
        def __init__(self, progress_writers=None, summary_period=None):
            self.training_updates = []
            self.training_summaries = []
            self.test_summaries = []
            coll_period = progress_writers[0].freq if (progress_writers and progress_writers[0]) else \
                          summary_period if summary_period is not None else \
                          sys.maxsize
            super(Function._ProgressCollector, self).__init__(coll_period, 0, sys.maxsize, 0, sys.maxsize, 0)
            self.__disown__()
        def on_write_training_update(self, samples, updates, aggregate_loss, aggregate_metric):
            aggregate_loss        = aggregate_loss[1]   - aggregate_loss[0]
            aggregate_metric      = aggregate_metric[1] - aggregate_metric[0]
            samples = samples[1]          - samples[0]
            aggregate_loss   /= (samples if samples != 0 else 1)
            aggregate_metric /= (samples if samples != 0 else 1)
            self.training_updates.append(Record(loss=aggregate_loss, metric=aggregate_metric, samples=samples))
        def on_write_test_update(self, *args, **kwargs):
            pass
        def on_write_training_summary(self, samples, updates, summaries, aggregate_loss, aggregate_metric, elapsed_milliseconds):
            aggregate_loss   /= (samples if samples != 0 else 1)
            aggregate_metric /= (samples if samples != 0 else 1)
            self.training_summaries.append(Record(loss=aggregate_loss, metric=aggregate_metric, samples=samples))
        def on_write_test_summary(self, samples, updates, summaries, aggregate_metric, elapsed_milliseconds):
            aggregate_metric /= (samples if samples != 0 else 1)
            self.test_summaries.append(Record(metric=aggregate_metric, samples=samples))
        def write(self, *args, **kwargs):
            pass

    def train(self, minibatch_source,
              minibatch_size=32, streams=None, model_inputs_to_streams=None, parameter_learners=[],
              callbacks=[], progress_frequency=None, max_epochs=None, epoch_size=None, max_samples=None):
        '''
        Trains a model, given by its criterion function, using the specified training parameters and configs.
        Different aspects of training such as data sources, checkpointing, cross validation, progress printing
        can be configured using the corresponding config classes.

        The input data can be specified as a data reader (:class:`~cntk.io.MinibatchSource`)
        for large corpora; or directly as numpy/scipy arrays if the data is so small that it
        is feasible to keep it all in RAM.

        Data is processed in minibatches. The minibatch size defaults to 32, which is a choice that commonly works well.
        However, for maximum efficiency, we recommend to experiment with minibatch sizes
        and choose the largest that converges well and does not exceed the GPU RAM.
        This is particularly important for distributed training, where
        often, the minibatch size can be increased throughout the training, which reduces data bandwidth
        and thus speeds up parallel training.

        If input data is given through a data reader (as opposed to directly as a numpy/scipy array),
        the user must also specify the epoch size. This is because data readers are used for
        large corpora, and the traditional definition of epoch size as number of samples in the corpus
        is not very relevant. Instead, CNTK really means the number of samples
        between summary actions, such as printing training progress, adjusting the learning rate, and/or checkpointing the model.

        The function returns an object that contains these members: `epoch_summaries` is a list that
        contains the progression of epoch loss (`.loss`) and metric (`.metric`) values and the corresponding
        number of labels (`.samples`) that they were averaged over. This is the same value that a progress printer would print as epoch
        summaries. `updates` is a similar list with the more fine-grained minibatch updates.
        If a `TestConfig` was specified, then `test_summary` is the metric and sample count on the specified test set
        for the final model.

        A number of callback mechanisms can optionally be specified as a list as `callbacks`.
        CNTK has a fixed set of callback types, and only those types are allowed in the `callbacks` list:
        An object of type :class:`~cntk.cntk_py.ProgressWriter` from :mod:`cntk.logging` is used for progress logging;
        a :class:`~cntk.train.training_session.CheckpointConfig` configures the checkpointing mechanism, which
        keeps copies of models at regular intervals and allows to seamlessly restart from a last checkpoint;
        a :class:`~cntk.train.training_session.TestConfig` allows to specify a test set that is evaluated at the end of the training;
        and a :class:`~cntk.train.training_session.CrossValidationConfig` specifies a user callback that can be used to adjust learning
        hyper-parameters or to denote to stop training, optionally based on a separate cross-validation data set.

        This is a convenience wrapper around :class:`cntk.train.trainer.Trainer` :class:`cntk.train.training_session.TrainingSession`.

        Args:
            self: the criterion function of a model to be trained. This is either a single-valued function (the loss)
             or a tuple-valued function (loss and metric).
            minibatch_source (:class:`~cntk.io.MinibatchSource` or tuple of numpy/scripy arrays):
             data source used for training. For large data, use a MinibatchSource. For small data, pass a tuple of numpy/scipy arrays.
             The number of streams/arrays must match the number of arguments of `self`.
            streams (tuple): (only if minibatch_source is a data reader) the streams of the minibatch_source in argument order.
             Not to be given if minibatch_source is specified as numpy/scipy arrays rather than a data reader.
            minibatch_size (int or :class:`~cntk.cntk_py.minibatch_size_schedule`, defaults to 32): minibatch size (or schedule) for training
            epoch_size (int): in CNTK, epoch size means the number of samples between outputting summary information and/or checkpointing.
             This must be specified unless the user directly passes numpy/scipy arrays for the `minibatch_source`.
            max_epochs (int, defaults to 1): maximum number of samples used for training; requires `epoch_size`
            parameter_learners (list): list of learners from :mod:`cntk.learners`
            callbacks (list): list of callback objects, which can be of type
             :class:`~cntk.cntk_py.ProgressWriter` from :mod:`cntk.logging` (for logging),
             :class:`~cntk.train.training_session.CheckpointConfig` (for check-pointing),
             :class:`~cntk.train.training_session.TestConfig` (for automatic final evaluation on a test set), and
             :class:`~cntk.train.training_session.CrossValidationConfig` (for cross-validation based training control).
             Except for progress writers, at most one of each is allowed.
            model_inputs_to_streams (dict): alternative to `streams`, specifying the mapping as a map from input variables to streams
            max_samples (int): maximum number of samples used for training; mutually exclusive with `max_epochs`
            progress_frequency (int): frequency in samples for aggregated progress printing. Defaults to `epoch_size` if given, or `None` otherwise

        Example:
         >>> # a simple logistic-regression model
         >>> N = 250
         >>> np.random.seed(0)
         >>> Y = np.random.randint(size=N, low=0, high=2)  # labels
         >>> X = (np.random.randn(N, 2)+3) * (Y[:,None]+1)   # data
         >>> # Our model expects float32 features, and cross-entropy expects one-hot encoded labels.
         >>> import scipy.sparse
         >>> Y = scipy.sparse.csr_matrix((np.ones(N,np.float32), (range(N), Y)), shape=(N, 2))
         >>> X = X.astype(np.float32)
         >>> model = cntk.layers.Dense(2, activation=None) # model function
         >>> import cntk.layers
         >>> @cntk.Function.with_signature(cntk.layers.Tensor[2], cntk.layers.SparseTensor[2]) # criterion function
         ... def criterion(data, label_one_hot):
         ...     z = model(data)  # apply model. Computes a non-normalized log probability for every output class.
         ...     return cntk.cross_entropy_with_softmax(z, label_one_hot)
         >>> learner = cntk.sgd(model.parameters, 0.1)
         >>> progress = criterion.train((X, Y), minibatch_size=25, max_epochs=2, epoch_size=125, parameter_learners=[learner])
         >>> print("%.2f" % progress.epoch_summaries[-1].loss) # get the final epoch's loss value
         0.68

        Returns:
         An object `progress` with `progress.epoch_summaries` and `progress.updates` being the progressions of av loss, av metric, and number of labels
          for epochs and updates (groups of minibatches), respectively. If a `TestConfig` was given, then `progress.test_summary`
          includes the result (.metric and .samples)
        '''
        if minibatch_size is None:
            raise ValueError("minibatch_size must not be None.")
        elif isinstance(minibatch_size, int): # convert to a schedule
            from ..train.training_session import minibatch_size_schedule
            minibatch_size = minibatch_size_schedule(minibatch_size)
        elif not isinstance(minibatch_size, cntk_py.minibatch_size_schedule):
            raise ValueError('minibatch_size must be an int or the result an call to the minibatch_size_schedule() function')
        # max_samples
        # Can be either directly specified as max_samples or indirectly as (max_epochs, epoch_size).
        if max_samples is None:
            # derive from (max_epochs, epoch_size)
            if epoch_size is None:
                from ..io import MinibatchSource, UserMinibatchSource
                if isinstance(minibatch_source, (MinibatchSource, UserMinibatchSource)): # UserMinibatchSource derives from cntk_py.SwigMinibatchSource, not MinibatchSource, for director purposes
                    raise ValueError("epoch_size must be specified, unless max_samples is given or input is given as numpy/scipy arrays.")
                first_input = _as_tuple(minibatch_source)[0]
                try:
                    epoch_size = len(first_input)
                except:
                    epoch_size = first_input.shape[0] # if input is csr_matrix
            if max_epochs is None:
                max_epochs = 1 # default to 1 epoch
            max_samples = int(max_epochs * epoch_size) # (we allow fractional epochs so our testing system can run abbreviated tests)
        elif max_epochs is not None:
            raise ValueError("max_epochs and max_samples are mutually exclusive.")
        # parse callbacks list into the 4 different parameters that training_session expects
        from ..train.training_session import training_session, CheckpointConfig, CrossValidationConfig, TestConfig
        from ..cntk_py import ProgressWriter
        configs = Record(progress_writers=[], checkpoint_configs=[None], cv_configs=[None], test_configs=[None])
        types_to_configs = {
            ProgressWriter:        configs.progress_writers,
            CheckpointConfig:      configs.checkpoint_configs,
            CrossValidationConfig: configs.cv_configs,
            TestConfig:            configs.test_configs
        }
        for cb in callbacks: # separate the callbacks list into one of 4 separate types
            for type, config in types_to_configs.items():
                if isinstance(cb, type):
                    if isinstance(cb, cntk.cntk_py.ProgressWriter): # multiple progress writers are allowed
                        config.append(cb)
                    elif config[0]:
                        ValueError('only one callback of type ' + str(type) + ' is permitted')
                    else:
                        config[0] = cb
            else:
                ValueError('callbacks list can only contain objects of type ProgressWriter, CheckpointConfig, CrossValidationConfig, and TestConfig.')
        # use a progress tracker to capture the loss, metric, and count values
        if progress_frequency is None and epoch_size is not None: # if epoch size is given then default training summary frequency to it
            progress_frequency = epoch_size
        collector = Function._ProgressCollector(configs.progress_writers, progress_frequency // minibatch_size[0] if progress_frequency is not None else None)
        # Trainer instance
        from ..train.trainer import Trainer
        trainer = Trainer(None, self, parameter_learners, progress_writers=configs.progress_writers + [collector])
        # input map
        if streams:
            if model_inputs_to_streams:
                raise ValueError("streams and model_inputs_to_streams are mutually exclusive.")
            model_inputs_to_streams = self.argument_map(*streams)
        # training session
        ts = training_session(trainer, minibatch_source, minibatch_size, model_inputs_to_streams=model_inputs_to_streams,
                              progress_frequency=progress_frequency, max_samples=max_samples,
                              checkpoint_config=configs.checkpoint_configs[0], cv_config=configs.cv_configs[0], test_config=configs.test_configs[0])
        ts.train()
        res = Record(updates=collector.training_updates, epoch_summaries=collector.training_summaries) if len(collector.training_summaries) > 0 else \
              Record(updates=[Record(loss=0, metric=0, samples=0)], epoch_summaries=[Record(loss=0, metric=0, samples=0)])
        if configs.test_configs[0]:
            res = res.updated_with(test_summary=collector.test_summaries[-1])
        return res

    def test(self, minibatch_source, minibatch_size=32, streams=None, model_inputs_to_streams=None, callbacks=None):
        '''
        Measures the performance of a model, given by its criterion function, in the form of
        average metric value (or loss if model has only one output) on a set of data.

        This is a convenience wrapper around :class:`cntk.eval.evaluator.Evaluator`.

        Args:
            minibatch_source (:class:`~cntk.io.MinibatchSource`): minibatch source for the test data
            minibatch_size (:class:`~cntk.cntk_py.minibatch_size_schedule` or int): minibatch size for evaluation
            streams (tuple): the streams of the minibatch_source in argument order
            model_inputs_to_streams (dict): mapping between input variables and input streams
            callbacks (progress writer or list of them): optionally, list of
             progress writers from :mod:`cntk.logging` to automatically track training
             progress.

        Returns:
         An object `test_summary` with `test_summary.metric` being the average metric, and `test_summary.samples` the number of labels in the test set.
        '''
        if minibatch_size is None:
            raise ValueError("minibatch_size must not be None.")
        # input map
        if streams:
            if model_inputs_to_streams:
                raise ValueError("streams and model_inputs_to_streams are mutually exclusive.")
            model_inputs_to_streams = self.argument_map(*streams)
        # wrap the data if needed
        from ..train.training_session import TrainingSession
        minibatch_source, model_inputs_to_streams = TrainingSession._sanitize_minibatch_source(minibatch_source, model_inputs_to_streams, self, infinitely_repeat=False)
        # use a progress tracker to capture the metric and count values
        collector = Function._ProgressCollector()
        # Evaluator instance
        from ..eval.evaluator import Evaluator
        outputs = self.outputs
        output = outputs[0] if len(outputs) == 1 else outputs[1] # use metric if present, otherwise loss
        # callbacks. Only ProgressWriter is allowed in test()
        from ..cntk_py import ProgressWriter
        if callbacks and any(not isinstance(cb, ProgressWriter) for cb in callbacks):
            ValueError('callbacks list must only contain objects of type ProgressWriter')
        progress_writers = callbacks or []
        evaluator = Evaluator(output, progress_writers + [collector])

        if minibatch_source.is_infinite():
            raise ValueError("minibatch_source must have a limited number of samples or sweeps.")
        # evaluation loop
        while True:
            data = minibatch_source.next_minibatch(minibatch_size) # fetch minibatch
            if not data:
                break                                              # until we hit the end
            evaluator.test_minibatch({ input: data[si] for input, si in model_inputs_to_streams.items()})
        evaluator.summarize_test_progress()
        return collector.test_summaries[-1]

    @typemap
    def save(self, filename, format=ModelFormat.CNTKv2, use_external_files_to_store_parameters=False):
        '''
        Save this function graph into a model file using the specified format.

        Use distributed.Communicator.is_main() to gate your call to save()
        in distributed environment.

        Args:
            filename (str): model path
            use_external_files_to_store_parameters (bool, optional): whether to save model parameters 
             to external files. This is for models larger than 2GB. Defaults to False.
        '''
        return super(Function, self).save(filename, format.value, use_external_files_to_store_parameters)

    @typemap
    def restore(self, filename):
        '''
        Restore the models parameters (in-place) from a saved model file

        Args:
            filename (str): saved model path

        Returns:
            `None`: this method only has the side-effect of loading the model parameters from the file
        '''
        return super(Function, self).restore(filename)

    @staticmethod
    def register_udf_deserialize_callback(op_name, callback):
        '''
        Register a callback function to be invoked when deserializing a user-
        defined function with the corresponding op name.

        When loading a model, CNTK will try to automatically reconstruct any
        (non-native) user-defined functions by invoking a static
        :func:`~cntk.ops.functions.UserFunction.deserialize` method of the
        corresponding UserFunction sub-class. This method allows to override
        default UDF deserialization behavior by specifying a user- defined
        function op name and the corresponding callback that should be invoked
        instead of the ``deserialize`` method.

        Args:
            op_name (str): unique op name of the user-defined function.
            callback (function): a function taking three arguments (a list of
             inputs to the UserFunction, a string name, and a state dictionary
             generated by the corresponding :func:`~cntk.ops.functions.UserFunction.serialize`
             method) and returns an instance of the user-defined function.
        '''
        if op_name in Function._udf_callback_map:
            raise ValueError("A callback for the UserFunction with op name {}"
                " has already been registered.".format(op_name));
        Function._udf_callback_map[op_name] = callback

    @staticmethod
    @typemap
    def load(model, device=None, format=ModelFormat.CNTKv2):
        '''
        Load the ``model``, that has been saved using :func:`~cntk.ops.functions.Function.save`.

        Args:
            model (str, bytes or bytearray): either a file path of a model file or a byte buffer
             containing the binary representation of a model.
            device (:class:`~cntk.device.DeviceDescriptor`, defaults to the current globally default device):
             specifies the device to allocate the model on.
            format (:class:`~cntk.ModelFormat`, defaults to CNTKv2 format): specifies the format of the file to load.
             if the specified format is ONNX, then model must be a filename.

        Returns:
            root node
        '''
        if not device:
            device = DeviceDescriptor.use_default_device()

        is_buffer = is_byte_buffer(model)

        is_file = False
        if not is_buffer:
            try:
                is_file = path.exists(model)
            except:
                pass

        if is_buffer:
            if format != ModelFormat.CNTKv2:
                raise ValueError('Loading from buffer only supported for CNTKv2 format.')
            return cntk_py.Function.load_from_buffer(model, device)

        if is_file:
            return cntk_py.Function.load(str(model), device, format.value)

        raise ValueError('Cannot load the model {} that is neither a file nor a byte buffer.'.format(model))

    @staticmethod
    def with_signature(*args, **kwargs):
        '''
        Decorator for defining a @Function with a given signature. Same as @Function followed by @Signature.

        Example:
         >>> from cntk.layers.typing import *
         >>> @Function.with_signature(Tensor[13])
         ... def f(x):
         ...     return x * x
         >>> print(f)
         ElementTimes(x: Tensor[13]) -> Tensor[13]
         >>> # which is equivalent to this:
         >>> @Function
         ... @Signature(Tensor[13])
         ... def f(x):
         ...     return x * x
         >>> print(f)
         ElementTimes(x: Tensor[13]) -> Tensor[13]

        '''
        def decorator(f):
            from cntk.layers.typing import Signature
            f = Signature(*args, **kwargs)(f)
            f = Function(f)
            return f
        return decorator

def BlockFunction(op_name, name):
    '''
    Decorator for defining a @Function as a BlockFunction. Same as @Function, but wrap the content into an :func:`~cntk.ops.as_block`.
    '''
    return lambda f: Function(f, make_block=True, op_name=op_name, name=name)

@typemap
def register_native_user_function(op_id, module_name, factory_method_name):
    '''
    Registers a native user-defined Function that can be subsequently instantiated
    using the 'native_user_function' method.

    Args:
        op_id (str): Unique id of the native user-defined Function to register.
         This id must be unique and an error will be reported if it matches
         the 'op_id' specified for any other registered native user-defined Function.
        module_name (str): Name of the module containing the factory method for creating
         instances of the native user-defined Function being registered. This is typically
         the name of a DLL/so which exports a factory method for creating instances of the
         native user-defined Function.
        factory_method_name (str): Name of the factory method for creating instances of the native
         user-defined Function being registered. This method must be an exported method of the
         specified module.
    '''
    return cntk_py.Function_register_native_user_function(op_id, module_name, factory_method_name)

@typemap
def native_user_function(op_id, operands, attributes=None, user_function_instance_name=''):
    '''
    Creates an instance of a user-defined Function previously registered using the
    'register_native_user_function' method.

    Args:
        op_id (str): Id of the native user-defined Function to instantiate.
         This must be the id that was used when registering the native user-function
         with the 'register_native_user_function' method.
        operands (list): input operands of the new instance of the native user-defined Function.
        user_function_instance_name (str): Name of the instance of the created native
         user-defined Function.

    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    if attributes is None:
        attributes = {}

    attributes = _py_dict_to_cntk_dict(attributes)
    return cntk_py.Function_native_user_function(op_id, operands, attributes, user_function_instance_name)

@typemap
def load_model(model, device=None, format=ModelFormat.CNTKv2):
    '''
    Alias for :func:`~cntk.ops.functions.Function.load`.
    '''
    return Function.load(model, device, format)

class UserFunction(Function):
    '''
    Base class of all user extension functions.

    If it has only one output, one can invoke Variable methods on it, which it
    will relay to its only output.

    Args:
        inputs (list): inputs to this function
        as_numpy (bool, optional): whether the data should be automatically
         converted from and to NumPy. Defaults to True. Specifying this as
         `False` passes the data as CNTK Value objects.
        name (str): name of this function
    '''

    def __init__(self, inputs, as_numpy=True, attributes=None, name=''):
        if  attributes is None:
            super(UserFunction, self).__init__(inputs, name)
        else:
            attributes = _py_dict_to_cntk_dict(attributes)
            super(UserFunction, self).__init__(inputs, attributes, name)
        self.set_native(False)
        self.as_numpy = as_numpy

        # Since the state will frequently not be used, we cache the None-state
        # to speed up.
        self._none_state =  cntk_py.UserBackPropState.create(self, cpu(), None)

        # Memory management for user defined functions has to be controlled by
        # the C++ side. For more information:
        # http://www.swig.org/Doc3.0/Python.html#Python_nn35
        self.__disown__()

    def _get_none_state(self, device=cpu()):
        if self._none_state.device() != device:
            self._none_state = cntk_py.UserBackPropState.create(self, device, None)

        return self._none_state

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
             A BackPropState instance, which is used by :func:`backward`.
        '''
        if self.as_numpy:
            inputs = self.inputs
            arguments = tuple(_value_as_sequence_or_array(v, inputs[i]) for i, v in enumerate(arguments))

        map_if_possible(outputs)
        map_if_possible(outputs_to_retain)

        args = arguments if len(arguments)>1 else arguments[0]

        if len(outputs) <= 1:
            state, result = self.forward(args, device, outputs_to_retain)
            for k in outputs:
                outputs[k] = result
        else:
            state = self.forward(args, outputs, device, outputs_to_retain)

        if isinstance(state, cntk_py.BackPropState):
            self._state_wrapped = False
        else:
            self._state_wrapped = True
            if state is None:
                state = self._get_none_state(device)
            else:
                state = cntk_py.UserBackPropState.create(self, device, state)

        if self.as_numpy:
            for k,v in outputs.items():
                if v is None:
                    raise ValueError('not all outputs have been provided')

                # FIXME: seq_starts
                outputs[k] = sanitize_batch(k, v, None, device)

        return state, outputs

    def _backward(self, state, root_gradients, variables):
        '''
        Backpropagates supplied ``root_gradients`` for one or more of the output
        variables of the Function, to calculate gradients with respect to
        ``variables``. Formally, multiplies the values of ``root_gradients`` by
        the Jacobian of the Function and returns the subset of the output that
        corresponds to ``variables``.

        This function calls :func:`backward`, which is to be implemented by the
        user.

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

        if self.as_numpy:
            map_if_possible(root_gradients)
            for v in root_gradients:
                if v.needs_gradient:
                    root_gradients[v] = _value_as_sequence_or_array(root_gradients[v], v)

        if not isinstance(state, cntk_py.BackPropState):
            raise ValueError('state must be of type BackPropState')

        if self._state_wrapped:
            state = cntk_py.UserBackPropState.data(state)

        map_if_possible(variables)

        if len(root_gradients) == 1:
            for rg in root_gradients.values():
                break
            root_gradients = rg

        if len(self.inputs) > 1:
            self.backward(state, root_gradients, variables)
        else:
            result = self.backward(state, root_gradients)
            for k in variables:
                variables[k] = result

        if self.as_numpy:
            for k, v in variables.items():
                if v is not None:
                    variables[k] = sanitize_batch(k, v, None, device)

    def _infer_outputs(self, outputs):
        outputs.extend(self.infer_outputs())

    def infer_outputs(self):
        '''
        Returns a list of all output variables this user-defined function
        outputs.

        Output variables are created by
        :meth:`~cntk.ops.output_variable`.
        '''
        raise NotImplementedError('infer_outputs has to be overwritten')

    def clone(self, cloned_inputs):
        '''
        Creates a clone of this user-defined function.

        It assumes that the constructor signature of the user's implementation
        of the user function takes the inputs as individual arguments followed
        by the operator name. If the signature is different, then this method
        needs to be overriden.

        Args:
            cloned_inputs: list of cloned inputs to the new user-defined
             Function clone to be created.

        Returns:
            A cloned instance of this user-defined function.
        '''
        return self.__class__(*cloned_inputs, name=self.name)

    def _serialize_impl(self):
        dictionary = _serialize(self)
        return _py_dict_to_cntk_dict(dictionary)

    @staticmethod
    def deserialize(inputs, name, state):
        '''
        A stub deserialize method for illustration purposes. User-defined functions
        need to provide their own implementation in order for CNTK to be able to
        reconstruct them when loading a model.

        Args:
            inputs (list): a list of inputs to the function
            name (str): name of this function
            state (dict): a state dictionary generated by the corresponding
             :func:`~cntk.ops.functions.UserFunction.serialize` method.

        Returns:
            An instance of the user-defined function.
        '''
        raise NotImplementedError('a stub method for illustration purposes.')

    @property
    def op_name(self):
        '''
        Unique operation name of this user-defined function.
        This property defaults to '<module>.<class>', but can be overridden.
        '''
        return self.__class__._op_name()

    def serialize(self):
        '''
        Generates a dictionary that captures the state of this user-defined function.

        This method must be overridden, if a user function has any state that needs
        to be preserved in the model dictionary.
        '''
        return {}

    @classmethod
    def _op_name(cls):
        return cls.__module__ + '.' + cls.__name__
