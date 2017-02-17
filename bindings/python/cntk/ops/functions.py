from cntk import cntk_py
from cntk.device import DeviceDescriptor
from cntk.utils import typemap, sanitize_var_map, sanitize_batch, \
        sanitize_dtype_cntk, value_to_seq, _as_tuple, variable_value_to_seq, Record
from cntk.utils.swig_helper import map_if_possible
from cntk.ops.variables import Variable
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
    '''

    # We override the constructors to implement an overload that constructs
    # a CNTK Functions from a Python function (@Function).
    def __new__(cls, *args, **kwargs):
        if len(args) > 0 and hasattr(args[0], '__call__') and not isinstance(args[0], Function): # overload
            return Function.to_Function(*args, **kwargs)
        return super(Function, cls).__new__(cls) # for some reason, passing *args, **kwargs fails with "object() takes no args

    def __init__(self, *args, **kwargs):
        if len(args) > 0 and hasattr(args[0], '__call__') and not isinstance(args[0], Function): # overload
            return
        super(Function, self).__init__(*args, **kwargs)

    class NamedOutput:
        def __init__(self, **kwargs):
            for kw in kwargs: # TODO: only allow one arg
                self.name = kw
                self.arg = kwargs[kw]

    _placeholders_under_construction = set()

    # helper to get the parameter names and annotations of a Python function
    @staticmethod
    def _get_param_names(f):
        # Note we only use non-optional params (assume any optional param is not specified).
        # This allows to, e.g., accept max(a, b, *more, name='') as a binary function
        import sys
        if sys.version_info.major >= 3:
            from inspect import getfullargspec
        else:
            def getfullargspec(f):
                from inspect import getargspec
                a = getargspec(f)
                return Record(args=a.args, varargs=a.varargs, varkw=a.keywords, defaults=a.defaults, kwonlyargs=[], kwonlydefaults=None, annotations={})
        param_specs = getfullargspec(f)
        annotations = param_specs.annotations
        arg_names = param_specs.args
        defaults = param_specs.defaults # "if this tuple has n elements, they correspond to the last n elements listed in args"
        if defaults:
            arg_names = arg_names[0:-len(defaults)]
        return (arg_names, annotations)

    # helper to create a CNTK placeholder or input for a given name
    # An input_variable is created if the parameter is annotated with a Tensor(...) type.
    # In this case, CNTK will immediately trigger type inference.
    # Unannotated parameters will yield placeholder_variables instead.
    @staticmethod
    def _make_arg_variable(name, annotations):
        from .. import placeholder_variable, input_variable
        from .variables import Variable
        if name in annotations and isinstance(annotations[name], Variable.Type):
            var_type = annotations[name]
            return input_variable(name=name, **var_type)
        else:
            return placeholder_variable(name=name)

    # construct a Function from a Python lambda
    # where the Function's input signature is defined by the lambda
    # Use this as a decorator, e.g.:
    #   @Function
    #   def f(x): return x * x
    # or with given shapes:
    #   @Function
    #   def f(x:Tensor(13)): return x * x
    # The latter form will create a CNTK Function over Inputs; the former over Placeholders.
    @staticmethod
    def to_Function(f, members = {}, make_block=False, op_name=None, name=None):
        from ..default_options import default_options
        # Parameter() creation inside code of a Function def is forbidden. Setting 'pure' blocks it in Parameter().
        with default_options(pure=True):
            f_name = f.__name__ # (for debugging)

            # get the parameter list through inspection
            arg_names, annotations = Function._get_param_names(f)

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
            args = [Function._make_arg_variable(name, annotations) for name in arg_names]

            # helpers
            ref_keeper = None  # BUGBUG: to work around the ref-counting issue with outputs
            def force_order_args(fun_args):
                from .. import plus, reduce_sum
                block_args = [Function._make_arg_variable(arg.name, annotations) for arg in fun_args]  # placeholders inside the BlockFunction
                combined_block_args = combine(block_args)                               # the content of the BlockFunction
                arg_map = list(zip(block_args, fun_args))                               # after wrapping, the block_args map to args
                combined_args = as_block(composite=combined_block_args, block_arguments_map=arg_map, block_op_name='Tuple')
                global ref_keeper   # TODO: should this really be 'nonlocal'?
                ref_keeper = combined_args    # BUGBUG workaround the ref-counting problem
                return combined_args.outputs
            def invoke(fun_args):
                try:
                    # hide Placeholders of this function from .signature() of any function defined inside
                    for arg in args:
                        Function._placeholders_under_construction.add(arg)
                    out = f(*fun_args)
                    if out is None:
                        raise TypeError("CNTK Function '{}' must return a value".format(f_name))
                    #if isinstance(out, Function): # a tuple member is wrapped in a NamedOutput class, we got a name for it
                    #    print('trying to print args for', f_name)
                    #    print('out args:', [arg.name for arg in out.arguments])
                finally:
                    # unhide Placeholders of this function again
                    for arg in args:
                        Function._placeholders_under_construction.remove(arg)
                # resolve tuples and NamedOutputs  --TODO: check for duplicates
                def resolve_named(output):
                    if isinstance(output, Function.NamedOutput): # a tuple member is wrapped in a NamedOutput class, we got a name for it
                        output = alias(output.arg, name=output.name)
                    elif isinstance(output, cntk_py.Variable):
                        output = combine([output]) # workaround: wrap in another combine() call
                    return output
                if isinstance(out, tuple): # multi-valued function, returned as a tuple
                    out = [resolve_named(output) for output in out]
                    # BUGBUG: combine() does not allow duplicates, so we wrap them in alias()
                    out_seen = set()
                    for i in range(len(out)):
                        out_i = out[i]
                        if out_i in out_seen:
                            out[i] = alias(out_i)
                            #print('alias-wrapping duplicate arg in', f_name)
                        else:
                            out_seen.add(out_i)
                    out = combine(out)  # --> turn into a combine()
                else:
                    out = resolve_named(out)
                return out
            # ensure parameter ordering
            # if called from BlockFunction() then wrap into a block
            if make_block: # if we make a block then run off a separate set
                block_args = [Function._make_arg_variable(arg.name, annotations) for arg in args]  # placeholders inside the BlockFunction
                out = invoke(block_args)
                out = as_block(composite=out, block_arguments_map=list(zip(block_args, args)), block_op_name=op_name, block_instance_name=name)
                #print('made block out of', f_name, op_name, name)
            # not a block
            else:
                fun_args = args
                #if len(fun_args) > 1:
                #    fun_args = force_order_args(fun_args)
                # BUGBUG: Python interpreter used to crash sometimes with this enabled, so for now fix it after the fact only if needed
                # now invoke the Python function
                out = invoke(fun_args)
                # BUGBUG workaround: fix it after the fact with an inefficient solution only if we got it wrong
                out_arg_names = [arg.name for arg in out.signature]
                if set(out_arg_names) == set(arg_names) and out_arg_names != arg_names:  # order came out wrong
                    #print('reexecuting function', f_name, 'because args came out as', out_arg_names, 'instead of', arg_names)
                    fun_args = force_order_args(fun_args)
                    out = invoke(fun_args)

            # verify that we got the parameter order right
            out_arg_names = [arg.name for arg in out.signature]
            assert out_arg_names == arg_names

            # BUGBUG: as_block() cannot *not* use an argument (e.g. temporarily changing a function to not use an input)
            if len(out.signature) != len(args):
                unfulfilled_args = set(out.signature) - set(args)
                if unfulfilled_args:
                    unfulfilled_arg_names = [arg.name for arg in unfulfilled_args]
                    raise TypeError("CNTK Function '{}' has {} missing arguments ({}), which is currently not supported".format(f_name, len(unfulfilled_arg_names), ", ".join(unfulfilled_arg_names)))
                else:
                    unused_args = set(args) - set(out.signature)
                    unused_arg_names = [arg.name for arg in unused_args]
                    raise TypeError("CNTK Function '{}' has {} unused arguments ({}), which is currently not supported".format(f_name, len(unused_arg_names), ", ".join(unused_arg_names)))

            # for debugging
            out.f_name = f_name  # keep in Python wrapper for debugging

            # add all members to the Python class
            # TODO: remove this, stuff should not be in the Python objects
            for key in members:   # UNTESTED
                out.__dict__[key] = members[key]
            return out

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
        determine the {placeholder: variable} map for use with various call operations
        Returns a dictionary from this function's placeholders to whatever arguments are passed.
        Accepted are both positional and keyword arguments.
        This mimics Python's argument interpretation, except that keyword arguments are not optional.
        This does not require the arguments to be Variables or Functions. It is also called by train_minibatch().
        '''
        params = self.signature    # function parameters
        if len(args) + len(kwargs) != len(params):
            raise TypeError("CNTK Function expected {} arguments, got {}".format(len(params), len(args)))

        # start with positional arguments
        arg_map = dict(zip(params, args))

        # now look up keyword arguments
        if len(kwargs) != 0:
            params_dict = { arg.name: arg for arg in params }
            for name, arg in kwargs.items():  # keyword args are matched by name
                if name not in params_dict:
                    raise TypeError("got an unexpected keyword argument '%s'" % name)
                param = params_dict[name]
                if param in arg_map:
                    raise SyntaxError("got multiple values for argument '%s'" % name)
                arg_map[param] = arg # add kw argument to dict
                param_uid = param.uid
                if isinstance(arg, (cntk_py.Variable, cntk_py.Function)): # for viewing in debugger
                    arg_uid = arg.uid
                    arg_name = arg.name
        assert len(arg_map) == len(params)

        return arg_map

    def update_signature(self, *arg_types, **kwarg_types):
        '''
        define input shapes, in-place
        e.g.
        model.update_signature(42)
        pass a list of objects that define the dimensions etc. of the placeholders
        Currently you can pass an int, a tuple, an Input, or a dict created with Type()
        '''
        arg_map = self.argument_map(*arg_types, **kwarg_types) # map type specs to Function parameters
        def to_input(arg_type, name):
            from cntk import input_variable
            from .variables import Variable
            if isinstance(arg_type, (int, tuple)): # just passed a shape
                return input_variable(shape=_as_tuple(arg_type), name=name)
            elif isinstance(arg_type, Variable.Type): # full type given as Tensor(...)
                return input_variable(name=name, **arg_type)
            else:
                raise TypeError("update_signature() expects arguments of type int, tuple of int, or Type.Variable")
        # map the given types:
        #  - create an Input with the given Type or shape
        #  - keep the name property of the Function parameter
        #  - skip argument types passed as None
        #  - TODO: should verify existing shape/axis information
        arg_map = { param: to_input(arg_type, name=param.name) for param, arg_type in arg_map.items() if arg_type is not None }
        self.replace_placeholders(arg_map)

    # TODO: add a back-compat version of update_signature with the beta name and accepting Input() variables instead.

    # TODO: change to tuple, or remove entirely
    class OrderedRecord(list):
        '''
        A container that behaves like a list and a class, in that the elements it stores
        can be accessed by an index or as a named class member.
        This is used as the return value of Function.__call__(numeric data)
        '''
        def __init__(self, item_list):
            for item in item_list:
                assert isinstance(item, tuple) and len(item)==2
            super(Function.OrderedRecord, self).__init__(item_list)
        def __getattr__(self, key):
            for item in self: # linear search for name; assuming it's faster than a map since these tuples only have a handful of items
                if item[0] == key:
                    return item[1]
            raise AttributeError("record has no attribute '{}'".format(key))
        def __setattr__(self, key, value):
            raise AttributeError('record is immutable')
        def __getitem__(self, key):
            return super(Function.OrderedRecord, self).__getitem__(key)[1]
        def __setitem__(self, key, value):
            raise AttributeError('record is immutable')
        def __iter__(self):
            class ValueIterator:
                def __init__(self, base_iter):
                    self.base_iter = base_iter
                def __iter__(self):
                    return self
                def __next__(self):
                    return self.base_iter.__next__()[1] # extract the values
            return ValueIterator(super(Function.OrderedRecord, self).__iter__())
        # __missing__, __iter__, __contains__, keys(), values(), __delitem__

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
             Ellipsis (...) will create a new Placeholder. E.g. plus(...,3) creates a new lambda that adds 3.

        Returns:
             In case of symbolic inputs, returns another CNTK Function object with inputs bound to the arguments.
             Otherwise returns an ordered record of numpy arrays for multi-output Functions, and a single numpy array otherwise.
        '''

        # parse argument list and map to the function's input
        arg_map = self.argument_map(*args, **kwargs)

        # if placeholders were excluded due to being under construction,
        # we must include them in the argmap, otherwise they will be cloned
        for arg in self.arguments:
            if arg not in arg_map:
                #print('excluded placeholder detected:', arg.name)
                arg_map[arg] = arg

        # determine whether this is eval() or clone()
        is_symbolic = any(isinstance(arg, (cntk_py.Function, cntk_py.Variable)) for arg in arg_map.values())

        # symbolic: return a cloned Function
        # applying the function means to inline its piece of graph
        if is_symbolic:
            return self.clone(CloneMethod.share, arg_map)

        # numeric: evaluate
        outputs = self.outputs
        _, output_map = self.forward(arg_map, outputs)
        assert len(output_map) == len(outputs)
        if len(output_map) > 1:
            return Function.OrderedRecord([(output.name, output_map[output]) for output in outputs])
        else: # single value: return numpy array and that's it
            return list(output_map.values())[0]

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
        Lastly, this also gives access to Functions and Variables inside this Function;s
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
            from ..graph import find_by_name
            item = typemap(find_by_name)(self.block_root if self.is_block else self, name)
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

    def __getitem__(self, arg):
        '''
        Slicing of a Function result.
        '''
        return self.output.__getitem__(arg)

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
        # C++ clone() can only clone composites. If we are not a composite, make it one using combine()
        if not self.is_composite:
            from cntk import combine
            #return combine([self]).clone(method, substitutions).root_function.arguments[0].owner
            # BUGBUG: This ^^ does not give me the correct .arguments, so we leave the extra combine() in for now.
            return combine([self]).clone(method, substitutions)

        method = getattr(cntk_py,
                'ParameterCloningMethod_' + CloneMethod(method).name.capitalize())
        substitutions = substitutions or {}
        if not isinstance(substitutions, dict):
            raise TypeError("Variable substitution map must be a dictionary")
        return super(Function, self).clone(method, substitutions)

    @property
    @typemap
    def constants(self):
        '''
        List of all `Constant` variables of this :class:`~cntk.ops.functions.Function`
        '''
        return super(Function, self).constants()

    def eval(self, arguments=None, device=None, as_numpy=True):
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
            device (:class:`~cntk.device.DeviceDescriptor`): the device descriptor that
             contains the type and id of the device on which the computation is
             to be performed.
            as_numpy (bool): whether to return the result as a NumPy array. Default True.
             Specifying this as False returns a CNTK Value which avoids a 
             costly conversion but returns a somewhat opaque object.

        Returns:
           dict or NumPy Array: Dict with keys of ouput variable names and values of
           output variable. A single NumPy array if there is only one output value.
        '''

        _, output_map = self.forward(arguments, self.outputs, device=device, as_numpy=as_numpy)

        if len(output_map) > 1:
            return output_map
        else:
            return list(output_map.values())[0]


    @typemap
    def forward(self, arguments, outputs, keep_for_backward=None, device=None, as_numpy=True):
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
            outputs (iterable): outputs to fetch values for.
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
             costly conversion but returns a somewhat opaque object.

        Returns:
             A tuple (BackPropState, map of outputs to NumPy arrays). The
             BackPropState is a handle taken by :func:`backward`.
        '''
        if device is None:
            device = DeviceDescriptor.use_default_device()

        in_var_map = sanitize_var_map(self.arguments, arguments,
                                      None, device)
        output_map = {v: None for v in outputs}
        keep_for_backward = set(keep_for_backward or {})

        state = super(Function, self)._forward(in_var_map, output_map, device,
                                             keep_for_backward)
        if as_numpy:
            for k in output_map:
                output_map[k] = variable_value_to_seq(output_map[k], k)

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
            var_gradients[var] = variable_value_to_seq(value, var)

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
             respect to all arguments that need gradient will be computed. If a variable
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
            wrt = [arg for arg in self.arguments if arg.needs_gradient]

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
        return super(Function, self).inputs(True)

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
            substitution (:class:`~cntk.ops.variables.Variable`): the variable
             that will replace the placeholder

        Returns:
            :class:`Function`: itself

        :raises ExceptionType: when the function has multiple placeholders.
        '''
        # TODO: was this removed on master, or never merged?
        #substitution = sanitize_substitution_var(substitution)
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
    def save(self, filename):
        '''
        Save this function graph into a model file using protobuf-based
        serialization.

        Args:
            filename (str): model path
        '''
        return super(Function, self).save_model(filename)

    def save_model(self, filename): # legacy name
        import warnings
        warnings.warn('This will be removed in future versions. Please use '
                'save(...) instead', DeprecationWarning)
        return self.save(filename)

    @typemap
    def restore(self, filename):
        '''
        Restore the models parameters (in-place) from a saved model file

        Args:
            filename (str): saved model path

        Returns:
            `None`: this method only has the side-effect of loading the model parameters from the file
        '''
        return super(Function, self).restore_model(filename)

    def restore_model(self, filename): # legacy name
        import warnings
        warnings.warn('This will be removed in future versions. Please use '
                'restore(...) instead', DeprecationWarning)
        return self.restore(filename)

    @staticmethod
    @typemap
    def load(filename, device=None):
        '''
        Load the model in ``filename``, that has been saved using
        :func:`~cntk.ops.functions.Function.save`.

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

@typemap
def load_model(filename, device=None):
    '''
    Alias for :func:`~cntk.ops.functions.Function.load`.
    '''
    return Function.load(filename, device)

@typemap
def save_model(model, filename): # legacy name
    import warnings
    warnings.warn('This will be removed in future versions. Please use '
            'model.save(...) instead', DeprecationWarning)
    return model.save(filename)


class UserFunction(Function):
    '''
    Base class of all user extension functions.

    If it has only one output, one can invoke Variable methods on it, which it
    will relay to its only output.

    '''
    def __init__(self, inputs, name=''):
        super(UserFunction, self).__init__(inputs, name)

        # Memory management for user defined functions has to be controlled by
        # the C++ side. For more information:
        # http://www.swig.org/Doc3.0/Python.html#Python_nn35
        self.__disown__()


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
        arguments = tuple(variable_value_to_seq(v, self.inputs[i]) for i, v in enumerate(arguments))

        map_if_possible(outputs)
        map_if_possible(outputs_to_retain)

        args = arguments if len(arguments)>1 else arguments[0]

        if len(outputs) <= 1:
            state, result = self.forward(args, device, outputs_to_retain)
            for k in outputs:
                outputs[k] = result
        else:
            state = self.forward(args, outputs, device, outputs_to_retain)

        if not isinstance(state, cntk_py.BackPropState):
            state = cntk_py.UserBackPropState(self, device, state)

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
            root_gradients[v] = variable_value_to_seq(root_gradients[v], v)
        map_if_possible(variables)


        if len(variables)>1:
            self.backward(cntk_py.UserBackPropState.data(state), root_gradients, variables)
        else:
            for rg in root_gradients.values():
                break
            result = self.backward(cntk_py.UserBackPropState.data(state), rg)
            for k in variables:
                variables[k] = result

        for k,v in variables.items():
            if v is None:
                raise ValueError('gradients were not provided for all variables')

            variables[k] = sanitize_batch(k, v, None, state.device())

    def _infer_outputs(self, outputs):
        outputs.extend(self.infer_outputs())

    def infer_outputs(self):
        '''
        Returns a list of all output variables this user-defined function
        outputs.

        Output variables are created by
        :meth:`~cntk.ops.functions.output_variable`.
        '''
        raise NotImplementedError('infer_outputs has to be overwritten')

    def clone(self, cloned_inputs):
        '''
        Creates a clone of this user-defined function.

        Args:
            cloned_inputs: list of cloned inputs to the new user-defined
             Function clone to be created.

        Returns:
            A cloned instance of this user-defined function.
        '''
        raise NotImplementedError('clone has to be overwritten')

    def op_name(self):
        '''
        Returns the operator name.
        '''
        return 'UserFunction'
