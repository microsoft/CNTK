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
    #   def f(x): x * x
    def __new__(cls, f, members = {}):
        from inspect import signature
        params = signature(f).parameters
        f_name = f.__name__
        from cntk import placeholder_variable, combine, alias
        args = [placeholder_variable(name=arg_name) for arg_name in list(params.keys())]
        #print("===============================", [arg for arg in list(params.keys())])
        # force them into the right order
        # Placeholders are ordered in depth-first traversal order.
        # By routing them through combine(), we force their traversal order to be first to last.
        # TODO: Get evidence that this is actually doing what it is meant to do.
        args = combine(args).outputs
        # execute the lambda with placeholders as inputs, which creates a piece of graph
        out = f(*args)
        # resolve NamedOutputs
        # TODO: check for duplicates
        def resolve_named(output):
            if isinstance(output, Function.NamedOutput): # a tuple member is wrapped in a NamedOutput class, we got a name for it
                output = combine([output.arg], name=output.name)
                #out = alias(output.arg, name=output.name)
                # BUGBUG: Fails with "ValueError: Variable(ElementTimes64_output) with unknown shape detected when compiling the Function graph!"
                #  TODO: verify that this is still the case. Either way, alias() is slow.
                # BUGBUG: Without alias, the names are not propagated into outputs.
                # BUGBUG: Forgetting [] in combine will hang combine().
            return output
        if isinstance(out, tuple): # multi-value function, returned as a tuple
            out = [resolve_named(output) for output in out]
        else:
            out = [resolve_named(out)]
        # implant the function name as the node name --TODO: should be op_name in a BlockFunction
        out = combine(out, name=f_name)
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
    def argument_map(self, *args, **kwargs):
        params = self._get_arguments()    # function parameters
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
        Currently you can pass either
        TODO: honor the names
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
        from .variables import Variable
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
        Forward function composition (other o self), same as Sequential([self, other]).
        '''
        # TODO: accept a tuple for other, e.g. projected LSTM:
        #   LSTM(500) >> (Dense(250), identity)
        # Input is assumed to be a tuple of the same number of elements.
        # Note: in Sequential(), don't start from identity, but from first item which can be a tuple.
        # Note: We can broadcast here, to implement Parallel(), e.g.
        #       identity >> (Recurrence(LSTM(500)), Recurrence(LSTM(500), go_backwards=True) >> splice
        # TODO: change splice() to accept a variable-number of arguments
        if isinstance(other, tuple):
            combine([f(self.outputs[i]) for i in range(len(other))])
        # TODO: Test this ^^
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

    _ = "(argument placeholder)" # pass Function._ to any expression to create a Scala-like lambda

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

               * dict: keys are input variable or names, and values are the input data.
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
            >>> C.combine([d.find_by_name('c')]).eval({a:[1], b:[2]})
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
