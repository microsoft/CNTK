from cntk import cntk_py
from cntk.device import DeviceDescriptor
from cntk.utils import typemap, sanitize_var_map, value_to_seq, _as_tuple
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

    BUGBUG: This ^^ is not correct. It is also the base of composites and multi-valued functions.
    '''

    class NamedOutput:
        def __init__(self, **kwargs):
            for kw in kwargs:
                self.name = kw
                self.arg = kwargs[kw]

    # construct a Function from a Python lambda
    # where the Function's input signature is defined by the lambda
    # Use this as a decorator, e.g.:
    #   @Function
    #   def f(x): return x * x
    def __new__(cls, f, members = {}):
        # get the parameter list, and also the function name, through inspection
        from inspect import signature, Parameter
        params = signature(f).parameters
        f_name = f.__name__
        arg_names = [name for name, param in params.items() if param.default == Parameter.empty] # only non-optional params become Placeholders
        # execute the lambda with placeholders as inputs, which creates a piece of graph
        from cntk import placeholder_variable, combine, alias, as_block
        args = [placeholder_variable(name=name) for name in arg_names]
        out = f(*args)

        # resolve NamedOutputs
        # TODO: check for duplicates
        def resolve_named(output):
            if isinstance(output, Function.NamedOutput): # a tuple member is wrapped in a NamedOutput class, we got a name for it
                output = combine([output.arg], name=output.name)
                #output = alias(output.arg, name=output.name)
                #output = plus(output.arg, 0, name=output.name)
                # BUGBUG: Fails with "ValueError: Variable(ElementTimes64_output) with unknown shape detected when compiling the Function graph!"
                #  TODO: verify that this is still the case. Either way, alias() is slow.
                # BUGBUG: Without alias, the names are not propagated into outputs.
                # BUGBUG: Forgetting [] in combine will hang combine().
            # BUGBUG: as_block() only accepts Functions, not Variables
            elif isinstance(output, cntk_py.Variable):
                output = combine([output]) # workaround: wrap in another combine() call
            return output
        if isinstance(out, tuple): # multi-value function, returned as a tuple
            out = [resolve_named(output) for output in out]
            out = combine(out)
        else:
            out = resolve_named(out)
        # BUGBUG: as_block() cannot *not* use an argument (e.g. temporarily changing a function to not use an input)
        if len(out.placeholders) != len(args):
            unused_args = set(args) - set(out.placeholders)
            unused_arg_names = [arg.name for arg in unused_args]
            raise TypeError("CNTK Function '{}' has {} unused arguments ({}), which is currently not supported".format(f_name, len(unused_arg_names), ", ".join(unused_arg_names)))

        # BEGIN WORKAROUND
        # force parameter order
        # BUGBUG: as_block() on the entire function is not really working, it looses names of its contents.
        #         As a workaround, wrap the args themselves into alias(), combine(), as_block().
        out_arg_names = [arg.name for arg in out.placeholders]
        if out_arg_names != arg_names   and False: # if order changed then force the order
            args1 = [placeholder_variable(name=name) for name in arg_names]
            combined_args = combine([alias(arg, arg.name) for arg in args1])
            args2 = [placeholder_variable(name=name) for name in arg_names]
            arg_map = list(zip(args1,args2))
            combined_args = as_block(combined_args, arg_map, f_name + '_parameter_ordering')
            # this now is a BlockFunction that maps all args to themselves, with forced ordering
            combined_args.replace_placeholders(list(zip(args2,args)))
            args = combined_args.outputs
            # and try it again
            out = f(*args)
            if isinstance(out, tuple): # multi-value function, returned as a tuple
                out = [resolve_named(output) for output in out]
                out = combine(out)
            else:
                out = resolve_named(out)
            out_arg_names = [arg.name for arg in out.placeholders]
            assert out_arg_names == arg_names
        # END WORKAROUND

        # wrap into a block as to ensure ordering of parameters
        # BUGBUG: clone() on identity() does not seem to work with this. Luckily we don't need BlockFunction for unary functions.
        #if len(out.placeholders) > 1: # skip for unary functions for now due to identity/clone bug
        ## BUGBUG: This looses names. So avoid as_block() entirely unless needed; hoping it will not be used for where it matters
        #    args2 = [placeholder_variable(name=name) for name in arg_names]
        #    arg_map = list(zip(args,args2))
        #    out = as_block(out, arg_map, f_name)

        # add all members to the Python class
        # TODO: This should really be a dictionary inside BlockFunction
        for key in members:   # UNTESTED
            out.__dict__[key] = members[key]
        return out

    def __init__(self, f, members = {}):
        # don't call the base class, since Function is abstract in C++
        pass

    # TODO: get by with placeholders only, do NOT replace with Input but rather Placeholder(shape).
    # TODO: move this inside argument_map()
    def _get_arguments(self):
        return [arg for arg in self.inputs if arg.is_input or arg.is_placeholder]

    # determine the {placeholder: variable} map for use with various call operations
    # Accepted are both positional and keyword arguments.
    # This mimics Python's argument interpretation, except that keyword arguments are not optional.
    # This does not require the arguments to be Variables or Functions. It is also called by train_minibatch().
    def argument_map(self, *args, **kwargs):
        params = self._get_arguments()    # function parameters
        #param_names = [param.name for param in params] # (debugging)
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
        #params = self._get_arguments()  # the function arguments to fill in
        #if len(arg_types) != len(params):
        #    raise TypeError("CNTK Function.update_signature() expected {} arguments, got {}".format(len(params), len(arg_types)))
        def to_input(arg, name):
            from cntk import input_variable
            if isinstance(arg, (int, tuple)): # just passed a shape
                return input_variable(shape=_as_tuple(arg), name=name)
            else:
                return input_variable(name=name, **arg)
        # map the given types:
        #  - create an Input with the given Type or shape
        #  - keep the name property of the Function parameter
        #  - skip argument types passed as None
        #  - TODO: should verify existing shape/axis information
        arg_map = { param: to_input(arg, name=param.name) for param, arg in arg_map.items() if arg is not None }
        self.replace_placeholders(arg_map)
        #for pair in zip(params, args):
        #    if pair[1] is not None: # passing None will not update the signature of this argument
        #        self.replace_placeholders({pair[0]: pair[1]})

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

    # TODO: if all inputs are actual data, this should eval() instead.
    # TODO: if passed an actual Python function, construct a Function from it. ...how does that affect the function signature??
    # TODO: accept a single tuple as args, for (F, G) >> plus. tuple members can be None = identity.
    def __call__(self, *args, **kwargs):
        '''
        Call a Function, either on symbolic or numeric inputs.

           * If at least one input is a CNTK Function or Variable, then
             return another CNTK Function object with inputs bound to the arguments.
             This is a short-hand for `f.clone(share, argument_map(*args, **kwargs))`.
           * Otherwise, all arguments must be numbers, numpy arrays, or a :class:`~cntk.io.MinibatchData` instance.
             Then perform the actual computation and return the numeric result.
             This is a short-hand for `f.eval(argument_map(*args, **kwargs))`,
             except that there is no `device` parameter. If you need that, use `eval()` directly.

        Args:
            *args, **kwargs: The arguments to pass to the Function.

        Returns:
             In case of symbolic inputs, returns another CNTK Function object with inputs bound to the arguments.
             Otherwise returns an ordered record of numpy arrays for multi-output Functions, and a single numpy array otherwise.
        '''

        # parse argument list and map to the function's input
        arg_map = self.argument_map(*args, **kwargs)

        # determine whether this is eval() or clone()
        #from .variables import Variable
        is_symbolic = any(isinstance(arg, (cntk_py.Function, cntk_py.Variable)) for arg in arg_map.values())

        # symbolic: return a cloned Function
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
        '''
        try:
            return self.__dict__[name]
        except KeyError:
            if len(self.outputs) == 1:
                return getattr(self.output, name)

        raise AttributeError("'%s' object has no attribute '%s'" %
                             (type(self), name))

    def dump(self):
        from ..graph import depth_first_search
        graph = depth_first_search(self.root_function, lambda x: not isinstance(x, cntk_py.Variable) or not x.is_output)
        names = dict()
        def name_it(item):
            if item.name != '':
                return item.name
            if item in names:
                name = names[item]
            else:
                def make_name(n): # come up with a letter sequence
                    if n < 26:
                        return chr(n + 97)
                    else:
                        return make_name(n // 26) + make_name(n % 26)
                name = make_name(len(names))
                names[item] = name
            #if item.name != '':
            #    name = name + '{' + item.name + '}'
            return name
        def print_item(item):
            name = name_it(item)
            if isinstance(item, cntk_py.Function):
                op_name = item.op_name
                #shape = list(output.shape for output in item.outputs)
                shape = '(' +  ', '.join([name_it(output) + ':' + "{}".format(output.shape) for output in item.root_function.outputs]) + ')'
                inputs = '(' +  ', '.join([name_it(input) + ':' + "{}".format(input.shape) for input in item.root_function.inputs]) + ')'
            #elif isinstance(item, cntk_py.Placeholder):
            #    op_name = "_"
            #    shape = item.shape
            #    inputs = ''
            elif isinstance(item, cntk_py.Constant):
                op_name = "Constant"
                shape = item.shape
                inputs = ''
            elif isinstance(item, cntk_py.Parameter):
                op_name = "Parameter"
                shape = item.shape
                inputs = ''
            elif isinstance(item, cntk_py.Variable):
                if item.is_parameter:
                    op_name = "Parameter"
                elif item.is_placeholder:
                    op_name = "Placeholder"
                elif item.is_input:
                    op_name = "Input"
                elif item.is_constant:
                    op_name = "Constant"
                else:
                    op_name = "Variable"
                shape = item.shape
                name = name + " " + item.uid
                inputs = ''
            print(' ', op_name, name, inputs, ':', shape)
            pass
        print(name_it(self))
        for item in graph:
            print_item(item)

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
        # normalize Function args to their Function.output variable
        # BUGBUG: This is a workaround, I think, since for other cases, this happens automatically.
        #         Without, SWIG throws "TypeError: cannot convert value of dictionary".
        #         This mapping should be removed once the TypeError has been fixed.
        substitutions = { param: (arg.output if isinstance(arg, Function) else arg) for param, arg in substitutions.items() }
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
            map of outputs to NumPy arrays; or a single NumPy array if Function has only one output
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
        df, f = self.forward(at, output, set(output), device)
        ones = {self.output: np.ones_like(v) for v in f.values()}
        grad_dict = self.backward(df, ones, unique_wrt)
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
    def block_composite(self):
        '''
        Returns the composite function underlying this block Function.
        Throws an exception of this is not a block Function.
        '''
        return super(Function, self).block_composite()

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
        starting from this node. Throws an exceptoin if ``name`` occurs
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
        this node. Throws an exceptoin if ``name`` occurs multiple times. If
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
        Save this function graph into a model file using protobuf-based serialization.

        Args:
            filename (str): model path
        '''
        return super(Function, self).save_model(filename)

    def save_model(self, filename): # legacy name
        return self.save(filename)

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

    @staticmethod
    @typemap
    def load(filename, device=None):
        '''
        Load the model in ``filename``, that has been saved using
        `:func:save_model`.

        Args:
            filename (str): filename to load the model from
            device (:class:`~cntk.DeviceDescriptor`, default is the default device):
             instance of DeviceDescriptor

        Returns:
            root node
        '''
        if not device:
            device = DeviceDescriptor.use_default_device()
        function = cntk_py.Function.load_model(filename, device)
        return function

@typemap
def load_model(filename, device=None):
    '''
    Alias for `Function.load`.
    '''
    return Function.load(filename, device)
