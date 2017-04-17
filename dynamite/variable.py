import numpy as np
import cntk  # note: keep in 'cntk' namespace in here
import collections
#from timeit import default_timer as timer

# some global settings we can control from outside, e.g. for debugging
class VariableGlobalConfig:
    use_batching = True
    use_coroutines = True
    enable_tracing = False
    use_arena_allocator = False

# TODO: move to contrib/dynamite/variable.py ; import .tensor_ops

# some global settings for playing with different configs:

# Functions vs Variables:
#  - CNTK dynamite has no functions!!! (other than Python lambdas)
#  - forward:
#     - raw model: model(p1, p2, p3, ..., x1, x2)
#     - bind the parameters -> model'(x1, x2) = model(P1, P2, P3, ..., _, _)
#     - apply to data -> model'' = model'(X1, X2) = a single value; not a function at all!
#     - loss = model''.value     # triggers lazy evaluation; but really, it is already fully determined
#  - backward:
#     - chain rule: take gradient from top and multiply with gradient of node -> grad_times(set_of_params, error_signal=1)
#     - (dp1, dp2, dp3, ...) = model''.grad_times_{p1,p2,p3}(e)    # where e=1.0 in normal backprop
#     - hence, to compute the gradient, pick the node; choose e (typ. 1.0); and call node.grad_times({ p1, p2, p3 }, e)
#     - ...how do we tell it where to place the gradients? It must connect to the V2 gradient NDArrayViews. Ah! Parameter Variables carry it from the start.
#     - grad_times() will trigger batch transformation; then read off the gradient functions from the *batched* graph (; then reoptimize? try it out)

# TODO:
#  - move the entire stuff into Variable?? Then create outside overloads, e.g. times = Variable.__matmul__ instead of the other way round

# TODO: where to move these?
INFER = 0
times_initializer = "(times_initializer)" # (dummy object only looked at by its object identity)

# make sure something is a variable; numbers and Numpy arrays get converted here
# We also short-circuit aliases.
def sanitize_input(x):
    return Constant(x) if not isinstance(x, Variable) else \
           x.inputs[0] if x.op is Variable._op_alias  else \
           x

# a little helper to pass 'additional_kwargs' easily
def as_kwargs(**kwargs):
    return kwargs

##############################################################################
#
# Dynamite Variable type
#
##############################################################################

class Variable:

    # ------------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------------

    generation_counter = 0

    def __new__(cls, shape, op, inputs, backprop_to_functions=None, additional_args=(), additional_kwargs=None):
        v = object.__new__(cls)
        v.shape = shape
        v.op = op
        v.inputs = tuple(sanitize_input(input) for input in inputs)
        v.backprop_to_functions = backprop_to_functions  # tuple of fun(v, g) -> g * dv/dinp_i
        v.additional_args   = additional_args
        v.additional_kwargs = additional_kwargs
        for inp in v.inputs:
            assert isinstance(inp, Variable)
        #v.needs_gradient = True
        v.computed = False
        v.generation_id = Variable.generation_counter # unique id, also useful for topo-sort
        Variable.generation_counter += 1
        return v

    # construct by cloning another
    def clone(self):
        assert not self.computed # (for now no need to clone after values have been computed already)
        assert not hasattr(self, 'initializer')
        res = Variable(self.shape, self.op, self.inputs, backprop_to_functions=self.backprop_to_functions, additional_args=self.additional_args, additional_kwargs=self.additional_kwargs)
        self.generation_id = Variable.generation_counter # must get a new unique id
        Variable.generation_counter += 1
        return res

    # overwrite an existing Variable in-place with a new one--used by make_batched_inputs() to keep existing references alive
    def replace_with(self, other):
        assert not self.computed  # TODO: is it possible that all or some have been computed at this point? Maybe some?
        assert not hasattr(self, 'initializer')
        self.shape                 = other.shape
        self.op                    = other.op
        self.inputs                = other.inputs
        self.backprop_to_functions = other.backprop_to_functions
        self.additional_args       = other.additional_args
        self.additional_kwargs     = other.additional_kwargs
        self.computed              = other.computed
        if self.computed:
            self.data              = other.data
        self.generation_id = Variable.generation_counter # BUGBUG: This screws up the idea of the gen id for topo-sort... :(
        Variable.generation_counter += 1

    # ------------------------------------------------------------------------
    # logging
    # ------------------------------------------------------------------------

    def type_as_char(self):
        return "P" if isinstance(self, Parameter)             else \
               "C" if isinstance(self, Constant)              else \
               "G" if self.op is cntk.NDArrayView.__getitem__ else \
               "S" if self.op is cntk.NDArrayView.splice      else \
               "V"

    def type_as_string(self, expand_getitem=False):
        if self.op == cntk.NDArrayView.__getitem__ and expand_getitem:
            t = self.inputs[0].type_as_string() + '[' + str(self.additional_args[0]) + ']'
        else:
            t = self.type_as_char()
            t += "{:05}".format(self.generation_id)
            if self.computed:
                t += '*'
        t += ':' + str(self.shape)
        return t

    def op_as_string(self):
        t = str(self.op)
        ts = t.split(' ')
        if ts[0] == '<function': # e.g. "<function NDArrayViewOpsMixin.__getitem__ at 0x000000453A649158>"
            t = ts[1]
        return t

    def signature_as_string(self, expand_getitem=False):
        t = self.type_as_string(expand_getitem=expand_getitem)
        t += " = " + self.op_as_string()
        if self.inputs:
            t += " (" + ", ".join([inp.type_as_string(expand_getitem=expand_getitem) for inp in self.inputs])
        if self.additional_args:
            t += '; ' + str(self.additional_args)
        if self.additional_kwargs:
            t += '; ' + ', '.join(name + '=' + str(val) for name, val in self.additional_kwargs.items())
        if self.inputs:
            t += ")"
        return t

    def dump_graph(self, skip_free=True):
        dump_graph([self], skip_free)

    def __repr__(self):
        return self.signature_as_string()

    # ------------------------------------------------------------------------
    # execution, node level
    # ------------------------------------------------------------------------

    # create a batched version of this, taking all shape etc. considerations into account
    #  - batched_inputs: for each arg of self, batch_size batch of args of operations in the batch
    #  - batch_size: number of operations that got batched
    def create_batched(self, batched_inputs, batch_size):
        assert batch_size > 1
        # create a new node for the batched op
        # In some cases, neither input got batched. In that case, just execute a single op and distribute its output
        shape_batched = (batch_size,) + self.shape
        op = self.op
        # if the operation is a reduction to (), we must modify it to not reduce over the batch axis
        # All ops in unary_reduction_ops are single-arg ops and are meant to accept an additional reduce_to_shape argument.
        # TODO: Do this with a per-operation function.
        if op in unary_reduction_ops:
            # E.g. (44,5) --> (44,); or two reductions in parallel, (2,44,25) --> (2,25).
            reduction_op = op
            def reduce_batch(arg):
                input_shape = arg.shape
                num_ones_to_insert = len(input_shape) - len(self.shape) - 1
                assert num_ones_to_insert > 0
                reduce_via_shape = (batch_size,) + (1,) * num_ones_to_insert + self.shape # intermediate
                res = reduction_op(arg, reduce_to_shape=reduce_via_shape)
                res = res.reshape(shape_batched)
                return res
            return Variable(shape_batched, reduce_batch, batched_inputs, backprop_to_functions=self.backprop_to_functions)
        return Variable(shape_batched, op, batched_inputs, backprop_to_functions=self.backprop_to_functions, additional_args=self.additional_args, additional_kwargs=self.additional_kwargs)

    def _call_op(self, out=None):
        try:
          args = tuple(input.data for input in self.inputs)
          if out:
              if self.additional_kwargs:
                  data = self.op(*(args + self.additional_args), out=out, **self.additional_kwargs)
              else:
                  data = self.op(*(args + self.additional_args), out=out)
          else:
              if self.additional_kwargs:
                  data = self.op(*(args + self.additional_args), **self.additional_kwargs)
              else:
                  data = self.op(*(args + self.additional_args))
          if data.shape != self.shape:
              self.dump_graph()
              print(data.shape, self.shape)
          assert data.shape == self.shape # sanity check of shape inference
          return data
        except Exception: # (for catching stuff in the debugger; remove this)
          self.dump_graph()
          print('_call_op failure:', self.op, self.shape, tuple(input.shape for input in self.inputs))
          raise
        pass

    def _compute_data(self, out=None): # perform op and store result in a new field 'data'
        assert not self.computed
        self.data = self._call_op(out=out)
        self.computed = True

    # ------------------------------------------------------------------------
    # execution, graph-level operations
    # ------------------------------------------------------------------------

    _batch_eval_fn = None   # lambda to call to yield from current coroutine
    @staticmethod
    def set_yield(batch_eval_fn):
        prev = Variable._batch_eval_fn
        Variable._batch_eval_fn = batch_eval_fn
        return prev

    def get_value(self):  # return the NDArrayView--computed lazily at this point if needed
        if not self.computed:  # lazy computation (this is where all the difficult stuff will happen w.r.t. batching)
            if Variable._batch_eval_fn:
                Variable._batch_eval_fn(self) # delegate to task scheduler to eval us
                assert self.computed
            else:
                evaluate_graph([self])
        return self.data

    def to_ndarray(self):
        return self.get_value().to_ndarray()

    # ------------------------------------------------------------------------
    # gradients
    # ------------------------------------------------------------------------

    # create Variable that is the gradient of self w.r.t. a set of parameters, multiplied with error_signal
    def backprop_to(self, i):  # get backprop function for inputs[i]; each fun(v, g) -> g * dv/dinp_i
        if not self.backprop_to_functions:
            self.dump_graph()
            print('backprop_to() missing for', self.signature_as_string())
            raise NotImplementedError('backprop_to missing')
        return self.backprop_to_functions[i]
    def grad_times(self, set_of_params, error_signal=1):
        error_signal = sanitize_input(error_signal)
        return create_gradient_graph(self, set_of_params, error_signal)

    @staticmethod
    def _op_aggregate(*args, out=None): # NDArrayView operation to aggregate many inputs
        arg0 = args[0]  # for now assume that all share the same shape; use first for reference
        num_items = len(args)
        if num_items == 1:
            return arg0
        res = out or cntk.NDArrayView(shape=arg0.shape, data_type=arg0.dtype, device=arg0.device) # note: we may broadcast, so arg0.shape may be wrong; doesn't matter, we should have 'out' always anyway
        for i in range(num_items):
            if i == 0:
                res[:] = args[i] # assign
            else:
                res += args[i] # aggregate
        return res

    @staticmethod
    def _op_aggregate_place_items(*args, shape=None, keys=None, out=None):
        if out:
            assert shape == out.shape
            res = out
        else:
            res = cntk.NDArrayView(shape)  # create a zero of the right size
        res -= res  # zero it out (can we use .setvalue?)
        for key, arg in zip(keys, args):
            res[key] += arg         # backprop the slice there
        return res

    # ------------------------------------------------------------------------
    # operator overloads (infix ops)
    # ------------------------------------------------------------------------

    def __add__(self, other):
        return plus(self, other)
    def __sub__(self, other):
        return minus(self, other)
    def __mul__(self, other):
        return element_times(self, other)
    def __gt__(self, other):
        return greater(self, other)
    def __matmul__(self, other):
        return times(self, other)
    # operator overloads (reverse infix ops)
    def __radd__(self, other):
        return Variable.__add__(other, self)
    def __rsub__(self, other):
        return Variable.__sub__(other, self)
    def __rmul__(self, other):
        return Variable.__mul__(other, self)
    def __rgt__(self, other):
        return Variable.__gt__(other, self)
    def __rmatmul__(self, other):
        return Variable.__matmul__(other, self)

    # ------------------------------------------------------------------------
    # slicing, splicing, reshaping
    # ------------------------------------------------------------------------

    @staticmethod
    def _op_place_item(input, g, key, out=None): # computation of gradient of __getitem__(); BUGBUG: highly inefficient
        # BUGBUG: Cannot back-prop into a huge matrix I just sliced from; needs in-place semantics!!!
        if out:
            res = out
            res -= res  # create a zero of the right size
        else:
            res = input - input  # create a zero of the right size
        res[key] = g         # copy the slice there
        return res

    def __getitem__(self, key): # note: for now not fully supported
        # determine the output shape
        if isinstance(key, int):
            first_dim = ()
        elif isinstance(key, slice):
            # various combinations are not yet supported
            assert key.step is None or key.step == 1
            assert key.start >= 0 and key.stop >= 0
            first_dim = (key.stop - key.start,)
        else:
            assert false # so far unsupported key type
        shape = first_dim + self.shape[1:]
        # create the Variable; or return ourselves if we slice the whole thing anyway (it's a no-op); used by make_batched_input()
        if shape == self.shape:
            return self
        return Variable(shape, cntk.NDArrayView.__getitem__, (self,),
                        backprop_to_functions=(
                            lambda v, g: Variable(v.inputs[0].shape, Variable._op_place_item, (v.inputs[0], g,),
                                                  additional_args=(key,)),),
                        additional_args=(key,))

    def _op_alias(x): # the op for alias
        return x

    def _op_barrier(x, tag): # the op for alias --tag is only used for batching and otherwise ignored
        return x

    def barrier(self, tag):
        return Variable(self.shape, Variable._op_barrier, (self,), additional_args=(tag,), backprop_to_functions=(lambda v, g: g,))

    @staticmethod
    def splice(*args, axis=-1):
        if axis >= 0:
            ValueError('splice: axis >= 0 case not implemented for now')
        return Variable((len(args),) + (1,) * (-1 - axis) + args[0].shape,
                        cntk.NDArrayView.splice,
                        args,
                        backprop_to_functions=None if axis != -1 else tuple(lambda v, g, i=i: g[i].reshape(args[i].shape) for i in range(len(args))), # BUGBUG: wrong axis unless -1
                        additional_kwargs=as_kwargs(axis=axis))

    def reshape(self, shape):
        if shape == self.shape:
            return self
        return Variable(shape, cntk.NDArrayView.reshape, (self,), additional_args=(shape,))

    def reduce_sum_to_shape(self, shape):
        if shape == self.shape:
            return self
        return Variable(shape, cntk.NDArrayView.reduce_sum, (self,), additional_kwargs=as_kwargs(reduce_to_shape=shape))

class Parameter(Variable):
    def __new__(cls, shape, initializer=None):
        return Variable.__new__(cls, shape, 'Parameter', [])
    def __init__(self, shape, initializer=None):
        if initializer:
            self.initializer = initializer
        self.op = Parameter._initialize
        self.additional_args = (self,)
        if all(dim != INFER for dim in shape):
            self._compute_data()
    def _initialize(self): # meant to be called from _compute_data() and thusly to return a cntk.core.NDArrayView
        assert all(dim != INFER for dim in self.shape)
        if hasattr(self, 'initializer'):
            data = cntk.NDArrayView.random_uniform_float(self.shape, -0.05, +0.05) # BUGBUG: device?? precision==float32??
            del self.initializer
        else:
            data = cntk.NDArrayView.from_dense(np.zeros(self.shape, np.float32)) # BUGBUG: device?? precision??
        data.__class__ = data.__class__ = cntk.core.NDArrayView
        return data
    def share_data_from(self, other): # keep a reference to the other Parameter's NDArrayView object
        # TODO: also accept the parameter's gradient  --needs to expose this from C++ (or at least Python Parameter)
        #       But they are owned by the learner, aren't they? How to get them out?
        #       -- new method share_gradient_with(self, other)! We can propagate up whether we have a gradient, for mem sharing
        data = other.data
        data.__class__ = data.__class__ = cntk.core.NDArrayView
        self.shape = data.shape
        self.data = data  # NDArrayView
        self.computed = True
        if hasattr(self, 'initializer'):
            del self.initializer
    def resize(self, shape):
        self.shape = shape
        if all(dim != INFER for dim in shape):
            self._compute_data()

class Constant(Variable):
    def __new__(cls, data, initializer=None): # data: cntk.core.NDArrayView or number or np.ndarray
        if not isinstance(data, cntk.NDArrayView):
            data = cntk.NDArrayView.from_dense(np.array(data, np.float32)) # BUGBUG: device?? precision??
        v = Variable.__new__(cls, data.shape, 'Constant', [])
        v.data = data  # NDArrayView
        v.computed = True
        return v

##############################################################################
#
# operators
# TODO: move into class
#
##############################################################################

# helper to determine the shape of an elementwise operation
def elementwise_shape(a,b):
    shapeA = a.shape if isinstance(a, (np.ndarray, Variable)) else ()
    shapeB = b.shape if isinstance(b, (np.ndarray, Variable)) else ()
    rank = max(len(shapeA), len(shapeB))
    shapeA = (1,) * (rank - len(shapeA)) + shapeA;
    shapeB = (1,) * (rank - len(shapeB)) + shapeB;
    return tuple(max(dimA, dimB) for dimA, dimB in zip(shapeA,shapeB))

def binary_op(opcode, backprop_to_functions=None):
    def f(a,b):
        return Variable(elementwise_shape(a,b), opcode, (a,b), backprop_to_functions=backprop_to_functions)
    return f

#def reducing_binary_op(opcode): # (unused)
#    @BroadcastingBinary
#    def f(a,b):
#        return Variable((), opcode, (a,b))
#    return f

def unary_op(opcode, backprop_to_functions=None):
    def f(x):
        return Variable(x.shape, opcode, (x,), backprop_to_functions=backprop_to_functions)
    return f

unary_reduction_ops = set() # unary_reduction_ops must be treated specially in batched execution; we collect them here during startup

def unary_reduction_op(opcode, backprop_to_functions=None):
    unary_reduction_ops.add(opcode)
    def f(x):
        return Variable((), opcode, (x,), backprop_to_functions=backprop_to_functions)
    return f

def _op_times_backprop_to_0(v, g): # fun(v, g) -> g * dv/da
    # BUGBUG: Must do the same dance as for _op_times_backprop_to_1().
    a = g
    b = v.inputs[1]
    shapeA = a.shape
    shapeB = b.shape
    return Variable(v.inputs[0].shape, cntk.NDArrayView.dot_transpose, (g, v.inputs[1]))

def _op_times_backprop_to_1(v, g): # fun(v, g) -> g * dv/db
    # This is the gradient into the matrix, which is v.inputs[1].
    # Nasty! The matmul gradient is no longer nice with true vectors.
    a = v.inputs[0]
    b = g
    shapeA = a.shape
    shapeB = b.shape
    if len(shapeA) == 1:
        shapeA = (1,) + shapeA
        a = a.reshape(shapeA)
    if len(shapeB) == 1:
        shapeB = (1,) + shapeB
        b = b.reshape(shapeB)
    shapeC = (shapeA[1], shapeB[1])
    res = Variable(shapeC, cntk.NDArrayView.transpose_dot, (a, b))
    if res.shape != v.inputs[1].shape:
        res = res.reshape(v.inputs[1].shape)
    return res
    #return Variable(v.inputs[1].shape, cntk.NDArrayView.transpose_dot, (v.inputs[0], g))

def times(a,b):
    if isinstance(a, (int, float)) and a == 0: # HACK HACK! Otherwise, backprop cannot handle this special case. Will fail for non-zero scalars
        return 0
    if hasattr(b, 'initializer'):
        shape = (b.shape[0] if b.shape[0] != INFER else a.shape[0] if isinstance(a, (np.ndarray, Variable)) else 1,
                 b.shape[1])
        b.resize(shape)
    shapeA = a.shape if isinstance(a, (np.ndarray, Variable)) else (b.shape[0],)
    shapeB = b.shape if isinstance(b, (np.ndarray, Variable)) else ()
    if shapeA != ():  # this special case to allow "0" for initial_state --need to do this more nicely
        if not (1 <= len(shapeA) <= 2) or not (1 <= len(shapeB) <= 2):
            raise TypeError('times only supports matrices and vectors')
        if shapeA[-1] != shapeB[0]:
            raise TypeError('inner dimensions do not match')
    shapeC = ()
    if len(shapeA) == 2:
        shapeC = shapeC + (shapeA[0],);
    if len(shapeB) == 2:
        shapeC = shapeC + (shapeB[1],);
    return Variable(shapeC, cntk.NDArrayView.dot, (a,b), backprop_to_functions=(_op_times_backprop_to_0, _op_times_backprop_to_1))

def times_transpose(a,b):
    if hasattr(b, 'initializer'):
        shape = (b.shape[0],
                 b.shape[1] if b.shape[1] != INFER else a.shape[0] if isinstance(a, (np.ndarray, Variable)) else 1)
        b.resize(shape)
    shapeA = a.shape if isinstance(a, (np.ndarray, Variable)) else (b.shape[1],)
    shapeB = b.shape if isinstance(b, (np.ndarray, Variable)) else ()
    if not (1 <= len(shapeA) <= 2) or not (1 <= len(shapeB) <= 2):
        raise TypeError('times only supports matrices and vectors')
    if shapeA[-1] != shapeB[-1]:
        raise TypeError('inner dimensions do not match')
    shapeC = ()
    if len(shapeA) == 2:
        shapeC = shapeC + (shapeA[0],);
    if len(shapeB) == 2:
        shapeC = shapeC + (shapeB[0],);
    return Variable(shapeC, cntk.NDArrayView.dot_transpose, (a,b)) # note: no gradient for now

zero = Constant(0)
one = Constant(1)

def _unbroadcast(input, g): # reduce 'g' to shape of input; for gradients of broadcasting operations
    if g.shape != input.shape:
        g = g.reduce_sum_to_shape(input.shape)
    return g

# BUGBUG: None of these will correctly inverse-broadcast in backprop_to(); need a way to pass size (or the buffer altogether, same as for arena allocator)
plus          = binary_op(cntk.NDArrayView.__add__, backprop_to_functions=(lambda v, g: _unbroadcast(v.inputs[0], g),               lambda v, g: _unbroadcast(v.inputs[1], g)))
minus         = binary_op(cntk.NDArrayView.__sub__, backprop_to_functions=(lambda v, g: _unbroadcast(v.inputs[0], g),               lambda v, g: _unbroadcast(v.inputs[1], zero - g))) # TODO: need a proper negate operator, which is easy with alpha
element_times = binary_op(cntk.NDArrayView.__mul__, backprop_to_functions=(lambda v, g: _unbroadcast(v.inputs[0], g * v.inputs[1]), lambda v, g: _unbroadcast(v.inputs[1], g * v.inputs[0]))) # TODO: test this
greater       = binary_op(cntk.NDArrayView.greater)  #, backprop_to_functions=(lambda v, g: g, lambda v, g: g))


# TODO: all explicit calls here must be moved (1) into Variable for base functionality; and (2) to Layers with broadcasting wrapper

tanh    = unary_op(cntk.NDArrayView.tanh)
sigmoid = unary_op(cntk.NDArrayView.sigmoid, backprop_to_functions=(lambda v, g: g * (v * (one-v)),))
relu    = unary_op(cntk.NDArrayView.relu,    backprop_to_functions=(lambda v, g: g * (v > zero),))
exp     = unary_op(cntk.NDArrayView.exp,     backprop_to_functions=(lambda v, g: g * v,))  # TODO: double-check this
alias   = unary_op(Variable._op_alias,       backprop_to_functions=(lambda v, g: g,))



def _pad_unreduce(ref, g): # pad 'g' to the right number of axes for un-reducing in gradient of reduce_X_sum(); makes g broadcastable into ref.inputs[0].shape
    input_rank  = len(ref.inputs[0].shape)
    output_rank = len(ref.shape)
    num_reduced_dims = input_rank - output_rank
    assert num_reduced_dims > 0 # (something must have gotten reduced)
    padded_g_shape = g.shape + (1,) * num_reduced_dims # add 1-dims that got reduced away
    return g.reshape(padded_g_shape)
def _broadcast_to(ref, g): # broadcast g to shape of 'ref'
    return g.reduce_sum_to_shape(ref.shape)  # hack: when given a shape, reduce_sum can actually broadcast...
reduce_sum     = unary_reduction_op(cntk.NDArrayView.reduce_sum,     backprop_to_functions=(lambda v, g: _broadcast_to(v.inputs[0], _pad_unreduce(v, g)),))
reduce_log_sum = unary_reduction_op(cntk.NDArrayView.reduce_log_sum, backprop_to_functions=(lambda v, g: _pad_unreduce(v, g) * exp(v.inputs[0] - _pad_unreduce(v, v)),)) # df / dx = exp(x)/exp(f) = exp(x - f)  --TODO: check this

#softmax = unary_op(cntk.NDArrayView.softmax)  # TODO: define this as multiple operations, exp(z-reduce_log_sum(z))

def cross_entropy_with_softmax(output, label):
    #return reduce_log_sum(output) - times_transpose(label, output)
    return reduce_log_sum(output) - reduce_sum(label * output)
    # TODO: either turn this into a special ^^ operator, or allow shape to be passed to __mul__
classification_error = cross_entropy_with_softmax  # TODO... for now

##############################################################################
#
# execution
# TODO: move up once it works, then merge into class Variable
#
##############################################################################

# TODO: This is less trivial than it seems; need to double-check and test very carefully
# BUGBUG: It indeed seems to have an error. Can we self-check?
def topo_sort(roots: list):
    order = []

    visited = set() # [id(obj)] remembers every item that has ever been added to the work_list
    def traverse(node):
        if id(node) not in visited:
            visited.add(id(node))
            for input in node.inputs:
                traverse(input)
            order.append(node)
    for input in roots:
        traverse(input)
    assert len(order) == len(visited)

    # old routine without stack that sometimes gave wrong result; if we face stack problems, then fix this up
    #stack = roots.copy()
    #num_implanted = 0
    #while stack:
    #    p = stack.pop()
    #    for v in p.inputs:
    #        if id(v) in visited:
    #            continue
    #        if p:
    #            v.parent = p # once we emit the first one, we can emit its parent, too
    #            num_implanted += 1 # (sanity check only)
    #            p = None
    #        stack.append(v)
    #        visited.add(id(v))
    #    while p:  # no children (left) to process -> we can emit this and all parents that are ready
    #        order.append(p)
    #        q = getattr(p, 'parent', None)
    #        if q:
    #            del p.parent # clean up after ourselves (may not be needed)
    #            num_implanted -= 1 # (sanity check only)
    #        p = q
    ## some checks
    #assert num_implanted == 0
    assert len(order) == len(visited)
    seen = set()
    for node in order:
        for input in node.inputs:
            assert id(input) in seen # node must not be referenced as an input before it was seen
            #assert input.generation_id < node.generation_id # generation_ids are sorted   .. NOT! :(
        seen.add(id(node))
    #print(tuple(v.generation_id for v in order))
    return order

def print_graph_stats(vars, tag=''):
    from collections import Counter
    nodes = topo_sort(vars)
    stats = Counter(v.type_as_char() for v in nodes)
    num_params   = stats['P'] if 'P' in stats else 0
    num_consts   = stats['C'] if 'C' in stats else 0
    num_vars     = stats['V'] if 'V' in stats else 0
    num_getitems = stats['G'] if 'G' in stats else 0
    num_splices  = stats['S'] if 'S' in stats else 0
    num_spliced_items = sum(v.shape[0] for v in nodes if v.type_as_char() == 'S')
    total = num_params + num_consts + num_vars + num_getitems + num_splices
    print(tag, total, 'nodes,', (num_vars, num_splices, num_spliced_items, num_getitems), '(#compute, #splice, #spliced_items, #slice),', (num_params, num_consts), '(parameters, constants)')

# excecution
#  - prep: for all nodes,
#     - determine set of consumers for each node
#     - set not-ready-inputs counter to #inputs
#     - add any node with not-ready-children counter==0 to ready batched group
#  - select a batched group to execute
#     - e.g. largest (=largest chance of full util, while others may become fuller as a result)
#  - execute the batched group
#     - inserting reshuffling operation into dense tensor form if needed
#     - perform as one batched operation
#  - for each member of the batched op, check each consumer whether it is now ready; if so, move to ready set
#     - sort each one right away into its batched group
#     - this requires consumer sets for all nodes, and a not-ready-children counter
#  - delete the batched group
def transform_to_batched_ops(vars):
    nodes = topo_sort(vars)    # (it is possible to implement this without, just more complex)
    #print(tuple(v.generation_id for v in topo_sort(vars)))
    num_nodes = len(nodes)
    expected_num_ops = sum(1 for v in nodes if not v.computed)

    # management of batched operations
    ready_ops = dict()  # [key] -> list of Variables   --note: non-deterministic in Python!

    def add_ready(v):
        key = v.key
        if key not in ready_ops:
            ready_ops[key] = [v] # first entry: create
        else:
            ready_ops[key].append(v)

    num_compute_launches = 0
    num_gathers = 0

    # execute all operations in op_batch as one CUDA operation
    # The data needs to be gathered first (an optimized version would make the
    # scatter lazy and avoid it if possible).
    # This transforms parallel operations into a single op, as follows:
    #  - e.g. c_r = a_r + b_r
    #  - a = gather(a_r)
    #  - b = gather(b_r)
    #  - c = a+b
    #  - c_r <- scatter_r(c)   # overwrites c_r in_place
    # where scatter() is a slice_view().
    # Hence, the original nodes a_r, b_r, and c_r are still valid.
    # Specifically, references to c_r will still read out the same value; just from a modified graph.
    # As an optimization gather(scatter_r(x)) == x (assuming the same batch depth).
    # E.g. if we then compute e_r = c_r +_r d_r, with c_r = scatter_r(c),
    # then gather(c_r) = c.
    # Hence, in this case, the c_r are not actually computed, and are no longer referenced in the graph
    # from their original location. There may be references from elsewhere.
    # The c_r are slice_views. If they are used, they will be computed lazily, which in the case of
    # a slice_view does nothing but creating a little view structure if the input has already computed.
    # As for gradients, if c_r is still used elsewhere in the graph, then gradients will flow through
    # both c and the original c_r which is now a slice_view.
    # So removing the reference to c_r should not change anything.
    # BUGBUG: If a_r is used at multiple places, the current optimizer will not notice, and gather it multiple times.
    #         We should create a hash table that keeps splice ops and discovers repeated ones.
    #         Basically for all operations we generate here. Or maybe just put them back into ready_ops for reconsideration?
    # BUGBUG: Backprop into parameters does not correctly batch. That's an aggregate over matrix products,
    #         which we could pre-gather before the matrix product. Not happening now because they are of varying length.
    def transform_batched_op(op_batch):
        # TODO: discover order on all inputs; if any reverse and no non-reversed then
        #    op_batch = reversed(op_batch)
        v0 = op_batch[0]
        # barrier() is just a dummy to control batching
        if v0.op is Variable._op_barrier:
            for v in op_batch: # mutate to an alias, now that the barrier's work is done
                v.op = Variable._op_alias
                v.additional_args = ()
            return
        def reslicing(args): # helper to test whether we are re-splicing something previously sliced (then we can short-circuit it)
            # BUGBUG: This currently ignores the axis parameter.
            arg0 = args[0]
            return arg0.op is cntk.NDArrayView.__getitem__ and \
                 all(arg.op is cntk.NDArrayView.__getitem__ and
                     arg.inputs[0] is arg0.inputs[0] and
                     arg.additional_args[0] == i + arg0.additional_args[0]
                     for i, arg in enumerate(args)) # returns True if all inputs are consecutive views onto the same object (from a previous batching op)
        nonlocal num_compute_launches, num_gathers
        if len(op_batch) == 1: # short-circuit this to avoid unnecessary splice (...actually already taken care of the check for all inputs being the same)
            # if the operation is a splice itself, then reassess whether it is now short-circuitable (applies to the final loss aggregation)
            if v0.op == cntk.NDArrayView.splice and reslicing(v0.inputs):
                args = v0.inputs
                arg0 = args[0]
                i0 = arg0.additional_args[0]
                v_batched = arg0.inputs[0][i0:i0 + len(args)] # it may be a sub-range, so slice it (which is a no-op if nothing is sliced)
                v0.replace_with(v_batched)
                return
            num_compute_launches += 1
            return
        # __getitem__() is not an actual operation (just a view) and thus cannot be parallelized;
        # reshape() the same;
        # splice() should be excluded if it came out of this function; for now we exclude all
        if v0.op is cntk.NDArrayView.__getitem__ or v0.op is cntk.NDArrayView.reshape or v0.op is cntk.NDArrayView.splice:
            return
        # all ops are the same, so use the first as the reference
        for inp in v0.inputs:
            assert isinstance(inp, Variable)
        is_mul = v0.op is cntk.NDArrayView.dot or v0.op is cntk.NDArrayView.dot_transpose or v0.op is cntk.NDArrayView.transpose_dot
        left_batch_axis_is_last = v0.op is cntk.NDArrayView.transpose_dot
        if left_batch_axis_is_last:
            # here we need to batch along an axis added to the right, not to the left; not sure how to do that properly
            # For now, transpose_dot should only be generated by gradient, and already batched. If this ever fails, we should just exclude it from batching.
            raise NotImplementedError('batching for transpose_dot not implemented')
        # sparse can not be properly batched for now
        # BUGBUG: This must become part of the type, stored inside Variable not data
        #if (isinstance(v0.inputs[0], Variable) and v0.inputs[0].data.is_sparse()):
        #    for v in op_batch:
        #        #v._compute_data()  # non-batched for now
        #        num_compute_launches += 1
        #    return 
        # determine rank for new axis; we insert a new axis, and for that, all objects must use aligned indices
        def rank(input):
            return len(input.shape) if isinstance(input, (Variable, np.ndarray)) else 0
        ranks = tuple(rank(input) for input in v0.inputs)
        new_rank = 1 + (max(ranks) if not is_mul else ranks[0])
        # batch all inputs by adding a new batch axis
        # create a new node for batching each input
        num_batched_ops = len(op_batch)

        def make_batched_input(i, arg0): # returns either arg0 or a slice/splice
            args = tuple(v.inputs[i] for v in op_batch)
            assert arg0 is args[0]
            # --- cases in which all are identical
            # matrix product is special, in that the right argument is always shared in the batch and not applied element-wise
            if is_mul and i == 1:
                if not all(arg is arg0 for arg in args):
                    dump_graph(op_batch)
                assert all(arg is arg0 for arg in args)
                return arg0
            # check whether all inputs are the same (e.g. add a bias)--then don't batch
            elif all(arg is arg0 for arg in args): # all the same args: use the object itself, assuming broadcasting
                #print(' !match')
                return arg0
            elif arg0.op is Variable._op_alias and all((arg.op is Variable._op_alias and arg.inputs[0] is arg0.inputs[0]) for arg in args): # all aliases of the same thing
                # This is the case where an identical op exists in all batch items, which got batched into a single one,
                # and the original reference was patched to an alias to that single one.
                return arg0
            # --- are we re-splicing?
            elif reslicing(args):
                i0 = arg0.additional_args[0]
                return arg0.inputs[0][i0:i0 + len(args)] # it may be a sub-range, so slice it (which is a no-op if nothing is sliced)
            # --- case where we must splice (copy) the inputs
            else:
                # need to do actual splice
                axis = ranks[i] - new_rank # negative; this will insert a new axis
                res = Variable.splice(*args, axis=axis)
                if res.op == cntk.NDArrayView.splice: # if not, it was short-circuited, so don't count it
                    nonlocal num_gathers
                    num_gathers += 1
                return res

        num_inputs = len(v0.inputs)
        #print('start')
        batched_inputs = tuple(make_batched_input(i, arg0)
                               for i, arg0 in enumerate(v0.inputs))
        # Note: if across the unbatched operations, all respective args are the same, make_batched_input will return the first; that is, v0.inputs[i].

        num_compute_launches += 1
        if all(ib is i0 for ib, i0 in zip(batched_inputs, v0.inputs)): # all unbatched ops are identical: just do it once and redistribute
            # BUGBUG? This never triggers, actually. Do we have a case? Maybe only 0*R in RNNUnit?
            assert all(batched_inputs[i] is v0.inputs[i] for i in range(num_inputs)) # (just another way of restating the condition.. remove this)
            # since all are identical, clone v0
            v_batched = v0.clone()
            # and mutate the original ops into aliases of the shared one
            # (We mutate the original one as well to keep things regular. Can be removed in the future.)
            for i, v in enumerate(op_batch):
                v.replace_with(alias(v_batched))
        else:
            # create a batched operation that matches the v0
            v_batched = v0.create_batched(batched_inputs, num_batched_ops)
            # and mutate all ops into a slice views into the new batched op
            for i, v in enumerate(op_batch):
                v.replace_with(v_batched[i])
        for i, v in enumerate(op_batch):
            assert v.shape == v0.shape
    # end of transform_batched_op

    # initialization
    #  - determine set of consumers for each node
    #  - set not-ready-inputs counter to #inputs
    #  - add any node with not-ready-children counter==0 to ready batched group
    for p in nodes:
        if p.computed:
            continue
        def make_key(p):
            # special case for __getitem__ and reshape: they are free and can always run first
            if p.op is cntk.NDArrayView.__getitem__ or p.op is cntk.NDArrayView.reshape:
                return (1, p.op,) # 1 gives it highest priority as a batch op
            # special case for _op_barrier: they are free but should always run last
            if p.op is Variable._op_barrier:
                return (-1, p.op, p.additional_args) # -1 gives it lowest priority as a batch op
            # special case for matmul: right matrix must be identical to be batchable
            if p.op is cntk.NDArrayView.dot or p.op is cntk.NDArrayView.dot_transpose or p.op is cntk.NDArrayView.transpose_dot:
                return (0, p.op, (p.inputs[0].shape, id(p.inputs[1])))
            # batch if both op and input shapes are the same
            # Python slices are not hashable
            def make_hashable(arg):
                return tuple(make_hashable(arg1) for arg1 in arg) if isinstance(arg, tuple) else \
                       str(arg)                                   if isinstance(arg, slice) else \
                       arg
            additional_args_sanitized = p.additional_args
            additional_args_sanitized = tuple(make_hashable(arg) for arg in additional_args_sanitized)
            # Python dicts are not hashable, so make additional_kwargs into a tuple if given (yuk)
            additional_kwargs_tuplified = p.additional_kwargs
            if additional_kwargs_tuplified:
                additional_kwargs_tuplified = tuple(
                    (arg_name, make_hashable(additional_kwargs_tuplified[arg_name])) for arg_name in sorted(additional_kwargs_tuplified.keys())
                )
            return (0, p.op, additional_args_sanitized, additional_kwargs_tuplified, tuple(v.shape for v in p.inputs))
        p.key = make_key(p)
        dummy_set_key_test = { p.key } # verify that it is hashable
        # TODO: must also include the storage format in the key; do this in C++ version
        p.consumers = []
        p.non_ready_inputs = 0
        for v in p.inputs:
            if isinstance(v, Variable) and not v.computed:
                v.consumers.append(p) # (due to topo sort, v is already initialized)
                p.non_ready_inputs += 1
        if p.non_ready_inputs == 0: # create initial set of ready ops
            add_ready(p)  # a leaf that's ready to go: make it part of the initial ready set

    # execute as long as still anything pending
    batches_run = 0
    ops_run = 0
    while ready_ops:
        # select the largest ready batch size
        # Note: We use generation_id here to make it deterministic.
        key = max(ready_ops.keys(), key=(lambda key: (key[0], len(ready_ops[key]), ready_ops[key][0].generation_id)))
        op_batch = ready_ops[key]
        # execute it
        #print('batch of', len(op_batch), 'for op', key)
        transform_batched_op(op_batch)
        # BUGBUG: Does not consider the case of multiple consumers of the same variable; we will currently redo the gather op
        batches_run += 1
        # done with this one
        #  - for each member of the batched op, check each consumer whether it is now ready; if so, move to ready set
        #  - delete the batched group
        del ready_ops[key]  # remove from set before we add the newly ready ones
        for v in op_batch:
            #assert not v.computed
            #v.computed = True # value is available now
            ops_run += 1
            for p in v.consumers:
                assert p.non_ready_inputs > 0
                p.non_ready_inputs -= 1
                if p.non_ready_inputs == 0:
                    add_ready(p)

# main evaluation function
#  - evaluate a set of root variables
#  - with full automatic dynamic batching
import time
from operator import mul as mul_operator
from functools import reduce as functools_reduce
# TODO: find a nicer way of doing this automatically
ops_with_out = { cntk.NDArrayView.__add__, cntk.NDArrayView.__sub__, cntk.NDArrayView.__mul__, cntk.NDArrayView.greater,
    cntk.NDArrayView.dot, cntk.NDArrayView.dot_transpose, cntk.NDArrayView.transpose_dot,
    cntk.NDArrayView.sigmoid, cntk.NDArrayView.tanh, cntk.NDArrayView.relu, cntk.NDArrayView.exp,
    cntk.NDArrayView.reduce_sum, cntk.NDArrayView.reduce_log_sum, cntk.NDArrayView.splice,
    Variable._op_aggregate, Variable._op_aggregate_place_items, Variable._op_place_item
}
def evaluate_graph(vars):
    # transform the graph from raw (individual batch items) to the batched graph
    print_graph_stats(vars, 'RAW:    ')
    if VariableGlobalConfig.use_batching:
        transform_to_batched_ops(vars)
        print_graph_stats(vars, 'BATCHED:')
        #transform_to_batched_ops(vars) # verify that the second time does not change it any further
        #print_graph_stats(vars)
    #print('--- after another transform ---')
    #dump_graph(vars)
    #transform_to_batched_ops(vars)
    # now actually compute the transformed graph
    nodes = topo_sort(vars)
    num_nodes = len(nodes)
    # simple arena allocation
    # So far this makes no difference in runtime. Maybe the CUDA allocator is already sync-free?
    mem_offset = 0
    for p in nodes:
        if not p.computed:
            if VariableGlobalConfig.use_arena_allocator and p.op in ops_with_out:
                mem_size = functools_reduce(mul_operator, p.shape, 1)
                #print('allocating', mem_size, ' for shape', p.shape, 'node', p.op_as_string())
                p.mem_offset = mem_offset
                p.mem_size = mem_size
                mem_offset += mem_size
            else:
                p.mem_offset = -1 # no arena memory for this op
                #print('no arena allocation for node', p.op_as_string())
    #print('arena size', mem_offset)
    if mem_offset:
        arena = cntk.NDArrayView((mem_offset,))
    # execution
    num_ops = 0
    start = time.time()
    for p in nodes:
        if not p.computed:
            out = None
            mem_offset = p.mem_offset
            if mem_offset != -1:
                out = arena[mem_offset:mem_offset + p.mem_size]
                if out.shape != p.shape:
                    out = out.reshape(p.shape)
            p._compute_data(out=out)
            num_ops += 1
            if VariableGlobalConfig.enable_tracing:
                val = p.data.to_ndarray()
                print('--- trace', p.signature_as_string(expand_getitem=True))
                print(val)
        #print(p.data.to_ndarray())
    end = time.time()
    dummy = vars[0].to_ndarray()
    end2 = time.time()
    #print('{:.4f} ms in {} ops'.format((end-start) * 1000, num_ops))
    #print('{:.4f} ms in {} ops'.format((end2-start) * 1000, num_ops))
    #dump_graph(vars)
    #transform_to_batched_ops(vars) # this shows that transforming after actually computing is correct

# gradient
# This computes the gradient of a variable (e.g. criterion) w.r.t. a set of model parameters, times an error signal.
# Call this only once for all parameters, otherwise you will duplicate computation (no caching!).
# Inputs:
#  - v: variable whose gradient is to be computed
#  - parameters: list (set) of Parameter variables hanging off v to compute the gradient for
#  - error_signal: to back-propagate into the root
# Output:
#  - map [param] -> its gradient Variable
def create_gradient_graph(root, parameters, error_signal):
    # batch all ops
    # BUGBUG: This can only run once for now; so don't do it here
    #transform_to_batched_ops((root,))
    nodes = topo_sort([root])
    # determine set of nodes that depend on the parameters
    active_set = { id(p) for p in parameters }
    for node in nodes:
        if any(id(input) in active_set for input in node.inputs):
            active_set.add(id(node))
    if id(root) not in active_set:
        # root not depending on any parameter (gradient is 0; we could just return 0 for those cases, but it's likely an error)
        raise ValueError('grad_times: variable does not depend on any of the given parameters')
    # function to create the actual aggregation operation for each gradient
    def create_aggregate(args): # list[Variable] -> Variable:
        # BUGBUG: we must collate dyadic matrix aggregations into a single matmul (or can we rely on batching in forward?)
        # --> an aggregate operator, which inspects the input; should also separate different kinds of ops it seems,
        #     so maybe when use use a gradient, just call the transform function on it?
        # short-circuit singleton aggregate
        if len(args) == 1:
            return args[0]
        shape = args[0].shape
        args = list(reversed(args)) # we traversed backwards; undo that
        # split args into different types that we deal with separately
        place_item_args = [arg for arg in args if arg.op is Variable._op_place_item]
        # ...other special kinds, e.g. matrix products
        other_args =      [arg for arg in args if arg.op is not Variable._op_place_item] # all others
        assert len(place_item_args) + len(other_args) == len(args) # check that we got them all
        # deal with _place_item()
        if place_item_args:
            # we must make sure that all input slices are filled exactly once; then this is a splice operation
            place_item_args = [Variable(shape, Variable._op_aggregate_place_items,
                                        tuple(arg.inputs[1] for arg in place_item_args), # (input[0] is the self-reference that allows _op_place_item() to know the output shape)
                                        additional_kwargs=as_kwargs(shape=shape, keys=tuple(arg.additional_args for arg in place_item_args)))]
        # ...deal with other special kinds
        # _op_aggregate in the end
        args = place_item_args + other_args
        if len(args) == 1:
            return args[0]
        elif len(args) == 2:
            return args[0] + args[1]
        else:
            return Variable(shape, Variable._op_aggregate, tuple(args))
    # now build the graph backwards
    # This is the error backpropagation algorithm in 12 lines of Python.
    gradients = dict() # [node] -> list(node's incoming gradients) := error_signal * droot/dnode
    gradients[root] = [error_signal]
    g_used = set() # (sanity check only)
    for node in filter(lambda node: id(node) in active_set, reversed(nodes)):
        g = create_aggregate(gradients[node])
        g_used.add(id(node))
        # backprop into each child
        for i, input in enumerate(node.inputs):
            if id(input) in active_set:
                # TODO: need to specially handle __getitem__(), it's a splice
                backprop_to = node.backprop_to(i)
                input_g = backprop_to(node, g)
                if input.shape != input_g.shape:
                    dump_graph(input_g)
                    print('gradient shape', input_g.shape, "came back different from input's shape", input.shape, 'for input', i, 'of op', node.signature_as_string())
                assert input.shape == input_g.shape
                if input not in gradients:
                    gradients[input] = [input_g]
                else:
                    assert id(input) not in g_used # ensure traversal order is correct (it really should!)
                    gradients[input].append(input_g)
    # gather the results
    res = { p: create_aggregate(gradients[p]) for p in parameters } # if a parameter does not feed root, then there will be no gradient, we will fail here
    return res

def dump_graph(vars, skip_free=True): # vars can be a Variable or an iterable of Variables
    if isinstance(vars, Variable):
        vars = [vars]
    for node in topo_sort(vars):
        if not skip_free or node.type_as_char() in {'V', 'S'}:
            print(node.signature_as_string(expand_getitem=True))

from greenlet import greenlet # very lighweight coroutines

# apply f() to a batch of arguments, with automatic batch parallelism (coroutines)
def map_batch(f, batch_args):
    result = [] # results of map operation
    if VariableGlobalConfig.use_coroutines:
        # create a coroutine for each batch entry
        # (a real implementation would switch to coroutines only upon encountering a need for the first time)
        zipped_batch_args = tuple(zip(*batch_args))
        coros = [lambda args=args: f(*args) for args in zipped_batch_args] # (without the 'args=args', lambda will capture a reference to args, not its value; hence all see only its last value)
        # create the greenlet scheduler for the coroutines
        current_coro_index = None # index of current batch item/coroutine
        def yield_to_first(): # kick off the process by yielding to the first coroutine
            nonlocal current_coro_index
            current_coro_index = 0
            greenlets[current_coro_index].switch()
        def yield_to_next(): # yield to the next coroutine
            nonlocal current_coro_index
            current_coro_index += 1
            if current_coro_index == len(greenlets):
                current_coro_index = 0
            greenlets[current_coro_index].switch()
        def coro_wrapper(coro):
            def run_coro():
                res = coro()
                result.append(res)
                yield_to_next()
            return run_coro
        greenlets = [greenlet(coro_wrapper(coro)) for coro in coros]
        # now run the schedule
        pending_vars = []
        def yield_to_batch_eval(v): # facilitate yielded batch eval
            # this function gets called by Variable.get_value() if the value is not yet computed
            nonlocal pending_vars
            # this schedules a Variable for computation
            # As long as we keep getting called from different coroutines, just collect these.
            # Once all coroutines have requested (or terminated), launch a batch eval of all collected ones; then reenter.
            pending_vars.append(v)
            if current_coro_index+1 == len(greenlets): # BUGBUG: this is too simplistic; we need to carefully consider coroutines that terminate early
                assert len(pending_vars) == len(greenlets)
                evaluate_graph(pending_vars) # for now
                pending_vars = []
            yield_to_next()
        prev_yield_to_batch_eval = Variable.set_yield(yield_to_batch_eval) # enable yielded batch computation
        yield_to_first()
        Variable.set_yield(prev_yield_to_batch_eval) # disable yielded batch computation
    else: # vanilla form
        for args in zip(*batch_args):
            res = f(*args)
            result.append(res)
    return result

##############################################################################
#
# higher-level interfaces
#
##############################################################################

def train_minibatch(criterion, *batch_args):
    # for now, manually do the batch loop
    print('\n-------------------- batch of', len(batch_args[0]), '--------------------\n')
    # perform the parallel map
    crits = map_batch(criterion, batch_args)
    crits = tuple(crit[0] for crit in crits) # pick out first item as criterion --TODO: generalize this

    # sum up the ce values
    crits_batched = Variable.splice(*crits)
    crit = reduce_sum(crits_batched)
    return crit

# left-over
    #
    # TODO: move the tree_reduce function out from here
    f = plus
    def tree_reduce(f, args):
        n = len(args)
        if   n > 2:  return f(tree_reduce(f, args[:n//2]),tree_reduce(f, args[n//2:]))
        elif n == 2: return f(args[0],args[1])
        else:        return args[0]
    crit = tree_reduce(plus, crits)
    # the return value is not yet computed, but any access will trigger lazy computation
    return crit
