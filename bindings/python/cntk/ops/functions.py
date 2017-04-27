from cntk import cntk_py, Value
from cntk.device import DeviceDescriptor, cpu
from cntk.internal import map_if_possible, typemap, sanitize_var_map,\
                          sanitize_batch, sanitize_dtype_cntk, _as_tuple,\
                          sanitize_variable_value_dict,\
                          sanitize_Function_attributes,\
                          _value_as_sequence_or_array
from cntk.internal.utils import get_python_function_arguments, \
                                map_function_arguments, _py_dict_to_cntk_dict
from cntk.internal import UserFunctionDeserializer
from ..variables import Record, Variable
from enum import Enum, unique
from os import path
import warnings

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
      >>> from cntk import debugging
      >>> debugging.dump_signature(f)
      Function(x: Sequence[tensor]) -> Sequence[tensor]

    The above form creates a CNTK Function whose arguments are placeholder variables.
    Such a function can only be combined with other symbolic functions.

    To train a Function or pass data to it, you need to declare the types
    of the arguments. In this case, the @Function decorator creates a CNTK Function
    whose arguments are input variables.

    If you use Python 3, Functions with types are declared using Python annotation syntax, e.g.::

      @Function
      def f(x:Tensor[13]):
          return x * x

    If you are working with Python 2.7, use CNTK's `@:class:~cntk.layers.typing.Signature` decorator instead::

      >>> from cntk.layers.typing import *
      >>> @Function
      ... @Signature(Tensor[13])
      ... def f(x):
      ...     return x * x
      >>> debugging.dump_signature(f)
      Function(x: Tensor[13]) -> Tensor[13]

    ``make_block=True`` is an internal parameter used to implement `@:func:~cntk.layers.blocks.BlockFunction()`.
    If `BlockFunction()` passes `True``, then the result will be wrapped
    in :func:``~cntk.ops.as_block()``, using the supplied ``op_name`` and ``name`` parameters, which are otherwise ignored.
    '''

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
    def _to_Function(f, make_block=False, op_name=None, name=None):
        '''implements @Function decorator; see :class:`~cntk.layers.functions.Function`'''
        f_name = f.__name__ # (only used for debugging and error messages)

        # helper to create a CNTK placeholder or input for a given name
        # An input is created if the parameter is annotated with a Tensor(...) type.
        # In this case, CNTK will immediately trigger type inference.
        # Unannotated parameters will yield placeholder_variables instead.
        from .. import placeholder, input
        def make_arg_variable(name, annotations):
            from ..variables import Variable
            if isinstance(annotations.get(name, None), Variable._Type):
                var_type = annotations[name]
                return input(name=name, **var_type)
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

            # verify that we got the parameter order right
            out_arg_names = [arg.name for arg in out.signature]
            assert out_arg_names == arg_names

            if len(out.signature) != len(args):
                unfulfilled_args = set(out.signature) - set(args)
                if unfulfilled_args:
                    unfulfilled_arg_names = [arg.name for arg in unfulfilled_args]
                    raise TypeError("CNTK Function '{}' has {} missing arguments ({}), which is currently not supported".format(f_name, len(unfulfilled_arg_names), ", ".join(unfulfilled_arg_names)))
                else:
                    unused_args = set(args) - set(out.signature)
                    unused_arg_names = [arg.name for arg in unused_args]
                    raise TypeError("CNTK Function '{}' has {} unused arguments ({}), which is currently not supported".format(f_name, len(unused_arg_names), ", ".join(unused_arg_names)))

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
            raise TypeError("CNTK Function expected {} arguments, got {}".format(len(params), len(args) + len(kwargs)))
        params_dict = { arg.name: arg for arg in params }
        return map_function_arguments(params, params_dict, *args, **kwargs)

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
            from cntk import input
            from ..variables import Variable
            if isinstance(arg_type, (int, tuple)): # just passed a shape
                return input(shape=_as_tuple(arg_type), name=name)
            elif isinstance(arg_type, Variable._Type): # full type given as Tensor(...)
                return input(name=name, **arg_type)
            else:
                raise TypeError("update_signature() expects arguments of type int, tuple of int, or Type.Variable")
        # map the given types:
        #  - create an Input with the given Type or shape
        #  - keep the name property of the Function parameter
        #  - skip argument types passed as None
        #  - TODO: should verify existing shape/axis information
        arg_map = { param: to_input(arg_type, name=param.name) for param, arg_type in arg_map.items() if arg_type is not None }
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
                from cntk import input
                return input(arg)
        args = [to_input(arg) for arg in arg_types]
        self.replace_placeholders(dict(zip(placeholders, args)))


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
            return self.clone(CloneMethod.share, arg_map)

        # numeric: evaluate
        outputs = self.outputs
        _, output_map = self.forward(arg_map, outputs)
        assert len(output_map) == len(outputs)
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
        List of all input variables of the Function that are not of type Parameter or Constant
        '''
        return super(Function, self).arguments()

    @property
    @typemap
    def attributes(self):
        '''
        List of the attributes of the function
        '''
        return sanitize_Function_attributes(super(Function, self).attributes())

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
           dict or NumPy Array: Dict with keys of ouput variable names and values of
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
            >>> v = C.input(shape=(3,))
            >>> f = C.reciprocal(v)
            >>> _, fv = f.forward({v:[[1, 2, 4]]})
            >>> list(fv.values())[0]
            array([[ 1.  ,  0.5 ,  0.25]], dtype=float32)

        Example:
            >>> # Passing sparse values as one-hot with a vocabulary size of 5
            >>> vocab_size = 5
            >>> v = C.sequence.input(shape=(vocab_size,), is_sparse=True)
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
            >>> v = C.sequence.input(shape=(vocab_size,), is_sparse=True)
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

        output_map = {v: None for v in outputs}
        keep_for_backward = set(keep_for_backward or {})

        state = super(Function, self)._forward(in_var_map, output_map, device,
                                               keep_for_backward)
        if as_numpy:
            for k, val in output_map.items():
                output_map[k] = _value_as_sequence_or_array(val, k)

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
            >>> v = C.input(shape=(1,), needs_gradient=True)
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
        device = state.device()
        root_gradients = sanitize_var_map(self.outputs, root_gradients,
                                          None, device)

        var_gradients = dict((var, None) for var in variables)

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
            >>> x = C.input(shape=(1,), needs_gradient=True)
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
            >>> a = C.input(shape=1, name='i')
            >>> b = C.input(shape=1, name='i')
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
            >>> a = C.input(shape=1, name='a')
            >>> b = C.input(shape=1, name='b')
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

    @typemap
    def save(self, filename):
        '''
        Save this function graph into a model file using protobuf-based
        serialization.

        Use distributed.Communicator.is_main() to gate your call to save()
        in distributed environment.

        Args:
            filename (str): model path
        '''
        return super(Function, self).save(filename)

    def save_model(self, filename): # legacy name
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
        return super(Function, self).restore(filename)

    def restore_model(self, filename): # legacy name
        warnings.warn('This will be removed in future versions. Please use '
                'restore(...) instead', DeprecationWarning)
        return self.restore(filename)

    @staticmethod
    @typemap
    def load(model, device=None, udf_factory_callback_map=None):
        '''
        Load the ``model``, that has been saved using :func:`~cntk.ops.functions.Function.save`.

        Args:
            model (str or bytes): either a filepath of a model file or a byte buffer 
             containing the binary representation of a model.
            device (:class:`~cntk.device.DeviceDescriptor`, defaults to the current globally default device):
             specifies the device to allocate the model on.
            udf_factory_callback_map (dict, default is `None`): if the model contains any user-defined
             functions, CNTK will try to automatically reconstruct them by invoking a static
             ``deserialize`` method of the corresponding Function sub-class. This method takes three 
             arguments (a list of inputs to the function, a string name, and a state dictionary
             generated by the corresponding :func:`~cntk.ops.functions.UserFunction.serialize` method) and
             returns an instance of the user-defined function. This optional argument allows to override
             default UDF deserialization behavior by providing a map of user-function op names and 
             corresponding lambdas that should be invoked instead of the ``deserialize`` method.

        Returns:
            root node
        '''
        if not device:
            device = DeviceDescriptor.use_default_device()

        deserializer = UserFunctionDeserializer(udf_factory_callback_map)

        is_buffer = isinstance(model, type(b'')) and not isinstance(b'', str)
        is_buffer = is_buffer or isinstance(model, bytearray)

        is_file = False
        if not is_buffer:
            try:
                is_file = path.exists(model)
            except:
                pass

        if is_buffer:
            return cntk_py.Function.load_from_buffer(model, device, deserializer)
        
        if is_file:
            return cntk_py.Function.load(model, device, deserializer)
        
        raise ValueError('Cannot load a model that is neither a file nor a byte buffer.')

@typemap
def register_native_user_function(op_name, module_name, factory_method_name):
    '''
    Registers a native user-defined Function that can be subsequently instantiated
    using the 'native_user_function' method.

    Args:
        op_name (str): Name of the native user-defined Function to register.
         This name must be unique and an error will be reported if it matches
         the 'op_name' specified for a previously registered native user-defined Function.
        module_name (str): Name of the module containing the factory method for creating 
         instances of the native user-defined Function being registered. This is typically
         the name of a DLL/so which exports a factory method for creating instances of the
         native user-defined Function.
        factory_method_name (str): Name of the factory method for creating instances of the native
         user-defined Function being registered. This method must be an exported method of the
         specified module.
    '''
    return cntk_py.Function_register_native_user_function(op_name, module_name, factory_method_name)

@typemap
def native_user_function(op_name, operands, user_function_instance_name=''):
    '''
    Creates an instance of a user-defined Function previously registered using the
    'register_native_user_function' method.

    Args:
        op_name (str): Name of the native user-defined Function to instantiate.
         This name must be the name that was used when registering the native user-function 
         with the 'register_native_user_function' method.
        operands (list): input operands of the new instance of the native user-defined Function.
        user_function_instance_name (str): Name of the instance of the created native 
         user-defined Function.

    Returns:
        :class:`~cntk.ops.functions.Function`
    '''
    return cntk_py.Function_native_user_function(op_name, operands, user_function_instance_name)

@typemap
def load_model(model, device=None, udf_factory_callback_map=None):
    '''
    Alias for :func:`~cntk.ops.functions.Function.load`.
    '''
    return Function.load(model, device, udf_factory_callback_map)

@typemap
def save_model(model, filename): # legacy name
    warnings.warn('This will be removed in future versions. Please use '
            'model.save(...) instead', DeprecationWarning)
    return model.save(filename)

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
    def __init__(self, inputs, as_numpy=True, name=''):
        super(UserFunction, self).__init__(inputs, name)
        self.as_numpy = as_numpy

        # Since the state will frequently not be used, we cache the None-state
        # to speed up.
        self._none_state =  cntk_py.UserBackPropState(self, cpu(), None)

        # Memory management for user defined functions has to be controlled by
        # the C++ side. For more information:
        # http://www.swig.org/Doc3.0/Python.html#Python_nn35
        self.__disown__()

    def _get_none_state(self, device=cpu()):
        if self._none_state.device() != device:
            self._none_state =  cntk_py.UserBackPropState(self, device, None)

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

        if state is None:
            state = self._get_none_state(device)
        elif not isinstance(state, cntk_py.BackPropState):
            state = cntk_py.UserBackPropState(self, device, state)

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

            state = cntk_py.UserBackPropState.data(state)

        else:
            if not isinstance(state, cntk_py.BackPropState):
                raise ValueError('if as_numpy=False, state must be of '
                        'type BackPropState')

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
            for k,v in variables.items():
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

        Args:
            cloned_inputs: list of cloned inputs to the new user-defined
             Function clone to be created.

        Returns:
            A cloned instance of this user-defined function.
        '''
        raise NotImplementedError('clone has to be overwritten')

    def _serialize(self):
        dictionary = {}
        dictionary['class'] = self.__class__.__name__
        dictionary['module'] = self.__class__.__module__
        dictionary['op_name'] = self.op_name
        dictionary['state'] = self.serialize()
        return _py_dict_to_cntk_dict(dictionary)

    def serialize(self):
        '''
        Generates a dictionary that captures the state of this user-defined function.

        This method must be overridden, if a user function has any state that needs
        to be preserved in the model dictionary.
        '''
        return {}