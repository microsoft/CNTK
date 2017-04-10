import numpy as np
import cntk  # note: keep in 'cntk' namespace in here
import collections
from timeit import default_timer as timer

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

# BUGBUG:
#  - transformation no longer combines the sequences after the print(), when they should all be batchable
#    This is since commit 48b72d5b3925ca9661b3b975f6a4af13b2952648
#    Maybe it now optimnizes the whole thing again although stuff is already executed? Some dangling ref triggering that?
#    Are the root vars getting their computed flag?

# TODO:
#  - implement grad_times
#     - for now no in-place updates
#  - longer term:
#     - arena allocation
#        - after batch transformation, all shapes are known --> we know the size of the arena
#        - we can then allocate one massive chunk and create the 'data' members as slice views into it
#          (with the exception of the parameter gradient, for which we use the gradient memory shared into the Parameter object upfront if given)
#        - but that requires all tensor operations to take a target variable
#        - unify NumericOp and NumericOpInPlace by passing an output variable; cf. Numpy. Reduces the surface.
#  - move the entire stuff into Variable?? Then create outside overloads, e.g. times = Variable.__matmul__ instead of the other way round

INFER = 0
times_initializer = "(times_initializer)" # (dummy object only looked at by its object identity)

# convert an input to an NDArrayView if not yet (it may be an ndarray of a Number)
# TODO: do this early on when *creating* the Variable object, rather than when using it
#       Also, in our own code at least, convert it to Variable outside, so that we don't repeatedly convert the same thing over.
def to_data(input):
    return input.data if isinstance(input, Variable) else cntk.NDArrayView.from_dense(np.array(input, np.float32)) # BUGBUG: device?? precision??

def to_Variable(x):
    return x if isinstance(x, Variable) else Constant(x)

class Variable:
    def __new__(cls, shape, op, inputs, backprop_to_functions=None, additional_args=()):
        v = object.__new__(cls)
        v.shape = shape
        v.op = op
        v.inputs = tuple(to_Variable(input) for input in inputs)
        v.backprop_to_functions = backprop_to_functions
        v.additional_args = additional_args
        for inp in v.inputs:
            assert isinstance(inp, Variable)
        # TODO: capture the gradient functions for all inputs that need gradients (we also need a flag for that)
        #v.needs_gradient = True
        v.computed = False
        return v
    _batch_eval_fn = None   # lambda to call to yield from current coroutine
    @staticmethod
    def set_yield(batch_eval_fn):
        prev = Variable._batch_eval_fn
        Variable._batch_eval_fn = batch_eval_fn
        return prev
    def _call_op(self):
        try:
          #args = tuple(to_data(input) for input in self.inputs)
          args = tuple(input.data for input in self.inputs)
          data = self.op(*(args + self.additional_args))
          if data.shape != self.shape:
               print(data.shape, self.shape)
          assert data.shape == self.shape # sanity check of shape inference
          return data
        except Exception: # (for catching stuff in the debugger; remove this)
          print('_call_op failure:', self.op, self.shape, (input.shape for input in self.inputs))
          #return
          raise
        pass
    def compute_data(self): # perform op and store result in a new field 'data'
        assert not self.computed
        self.data = self._call_op()
        self.computed = True
    #@property
    def get_value(self):  # return the NDArrayView--computed lazily at this point if needed
        if not self.computed:  # lazy computation (this is where all the difficult stuff will happen w.r.t. batching)
            if Variable._batch_eval_fn:
                Variable._batch_eval_fn(self) # delegate to task scheduler to eval us
                assert self.computed
            else:
                batch_eval([self])
        return self.data
    def to_ndarray(self):
        return self.get_value().to_ndarray()
    # create Variable that is the gradient of self w.r.t. a set of parameters, multiplied with error_signal
    def backprop_to(self, i):  # get backprop function for inputs[i]
        if not self.backprop_to_functions:
            return lambda v, g: g # dummy for now, so that we can debug a partially implemented system
        return self.backprop_to_functions[i]
    def grad_times(self, set_of_params, error_signal=1):
        error_signal = to_Variable(error_signal)
        return create_gradient_graph(self, set_of_params, error_signal)
    # operator overloads
    def __add__(self, other):
        return plus(self, other)
    def __sub__(self, other):
        return minus(self, other)
    def __mul__(self, other):
        return element_times(self, other)
    def __matmul__(self, other):
        return times(self, other)
    #def __getitem__(self, key):
    #    assert isinstance(key, int)  # BUGBUG: for now; later must dupplicate the complex logic for interpreting key
    #    return Variable(self.shape[1:], cntk.NDArrayView.__getitem__, (self,), additional_args=(key,))
    def _op_alias(x): # the op for alias
        return x
    #def alias(self): # wrap an identity function around ourselves
    #    return Variable(self.shape, Variable._op_alias, (self,))
    @staticmethod
    def splice(*args):
        return Variable((len(args),) + args[0].shape, cntk.NDArrayView.splice, args)

class Parameter(Variable):
    def __new__(cls, shape, initializer=None):
        v = Variable.__new__(cls, shape, 'Parameter', [])
        return v
    def __init__(self, shape, initializer=None):
        if initializer:
            self.initializer = initializer
        self.computed = True # BUGBUG: but we don't have data
    def share_data_from(self, other): # keep a reference to the other Parameter's NDArrayView object
        # TODO: also accept the parameter's gradient  --needs to expose this from C++ (or at least Python Parameter)
        #       But they are owned by the learner, aren't they? How to get them out?
        #       -- new method share_gradient_with(self, other)! We can propagate up whether we have a gradient, for mem sharing
        data = other.data
        data.__class__ = data.__class__ = cntk.core.NDArrayView
        self.shape = data.shape
        self.data = data  # NDArrayView
    def resize(self, shape):
        self.shape = shape

class Constant(Variable):
    def __new__(cls, data, initializer=None): # data: cntk.core.NDArrayView or number or np.ndarray
        if not isinstance(data, cntk.NDArrayView):
            data = cntk.NDArrayView.from_dense(np.array(data, np.float32)) # BUGBUG: device?? precision??
        v = Variable.__new__(cls, data.shape, 'Constant', [])
        v.data = data  # NDArrayView
        v.computed = True
        return v

def BroadcastingBinary(binary_op):
    # BUGBUG: testing for nested sequences must know how to align... how to do that?
    def broadcasting_binary_op(a,b):
        if isinstance(a, list):
            if isinstance(b, list):
                return [broadcasting_binary_op(x,y) for x,y in zip(a,b)]
            return map(lambda sample: broadcasting_binary_op(sample, b), a)
        elif isinstance(b, list):
            return map(lambda sample: broadcasting_binary_op(a, sample), b)
        return binary_op(a,b)
    return broadcasting_binary_op

def elementwise_shape(a,b):
    shapeA = a.shape if isinstance(a, (np.ndarray, Variable)) else ()
    shapeB = b.shape if isinstance(b, (np.ndarray, Variable)) else ()
    rank = max(len(shapeA), len(shapeB))
    shapeA = (1,) * (rank - len(shapeA)) + shapeA;
    shapeB = (1,) * (rank - len(shapeB)) + shapeB;
    return tuple(max(dimA, dimB) for dimA, dimB in zip(shapeA,shapeB))

def binary_op(opcode):
    @BroadcastingBinary
    def f(a,b):
        return Variable(elementwise_shape(a,b), opcode, (a,b))
    return f

def reducing_binary_op(opcode): # (unused)
    @BroadcastingBinary
    def f(a,b):
        return Variable((), opcode, (a,b))
    return f

def unary_op(opcode, backprop_to_functions=None):
    def f(x):
        if isinstance(x, list): # broadcast along sequence
            return map(f, x)
        return Variable(x.shape, opcode, (x,), backprop_to_functions=backprop_to_functions)
    return f

unary_reduction_ops = set() # unary_reduction_ops must be treated specially in batched execution; we collect them here during startup

def unary_reduction_op(opcode):
    unary_reduction_ops.add(opcode)
    def f(x):
        if isinstance(x, list): # broadcast along sequence
            return map(f, x)
        return Variable((), opcode, (x,))
    return f

@BroadcastingBinary
def times(a,b):
    if hasattr(b, 'initializer'):
        b.resize((b.shape[0] if b.shape[0] != INFER else a.shape[0] if isinstance(a, (np.ndarray, Variable)) else 1,
                  b.shape[1]))
        del b.initializer
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
    return Variable(shapeC, cntk.NDArrayView.dot, (a,b))

@BroadcastingBinary
def times_transpose(a,b):
    if hasattr(b, 'initializer'):
        b.resize((b.shape[0],
                  b.shape[1] if b.shape[1] != INFER else a.shape[0] if isinstance(a, (np.ndarray, Variable)) else 1))
        del b.initializer
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
    return Variable(shapeC, cntk.NDArrayView.dot_transpose, (a,b))

plus = binary_op(cntk.NDArrayView.__add__)
minus = binary_op(cntk.NDArrayView.__sub__)
element_times = binary_op(cntk.NDArrayView.__mul__)

tanh = unary_op(cntk.NDArrayView.tanh)
one = Constant(1)
sigmoid = unary_op(cntk.NDArrayView.sigmoid, backprop_to_functions=(lambda v, g: g * (v * (one-v)),))
relu = unary_op(cntk.NDArrayView.relu)
#softmax = unary_op(cntk.NDArrayView.softmax)
#row_slice_0 = unary_op(cntk.NDArrayView.row_slice)
#def row_slice(x, begin, end):
#    return row_slice_0(x) # (ignore dims for this test)
reduce_log_sum = unary_reduction_op(cntk.NDArrayView.reduce_log_sum)
reduce_sum     = unary_reduction_op(cntk.NDArrayView.reduce_sum)

def cross_entropy_with_softmax(output, label):
    #return reduce_log_sum(output) - times_transpose(label, output)
    return reduce_log_sum(output) - reduce_sum(label * output)
    # TODO: either turn this into a special ^^ operator, or allow shape to be passed to __mul__
classification_error = cross_entropy_with_softmax  # TODO... for now

def identity(x):
    return x

def Model(**kwargs):
    def patch(f):
        f.__ismodel__ = True
        f.get_parameters = get_parameters  # BUGBUG: It's not this simple. Must create a class, it seems.
        for name, value in kwargs.items():
            setattr(f, name, value) # add all as class members
        #def mygetitem(self, x):
        #    x
        #f.__getitem__ = mygetitem
        return f
    return patch

def Dense(N, activation=identity, name=''):
    W = Parameter((INFER,N), initializer=times_initializer)
    b = Parameter((N,))
    @Model(W=W, b=b)
    def dense(x):
        return activation(x @ W + b)
    return dense

def LogValues(): # fake layer to print the value as it passes through; for testing direct-mode values/co-routines
    def log_values(x):
        x_cpu = x.to_ndarray() # force computation
        #print(x_cpu)
        return x
    return log_values

def RNNUnit(N, activation=sigmoid, name=''):
    W = Parameter((INFER,N), initializer=times_initializer)
    R = Parameter((N,N),     initializer=times_initializer)
    b = Parameter((N,))
    @Model(W=W, R=R, b=b)
    def rnn_step(s,x):
        return activation(plus(plus(times(s,R), times(x,W)), b))
    return rnn_step

def LSTM(N, activation=sigmoid):
    # TODO
    b  = Parameter((       3 * N,))                                 # bias
    W  = Parameter((INFER, 3 * N,), initializer=times_initializer)  # input
    R  = Parameter((N    , 3 * N,), initializer=times_initializer)  # hidden-to-hidden

    @Model(W=W, R=R, b=b)
    def lstm(dhc, x):
        dh, dc = dhc  # destructure the tuple
        # projected contribution from input(s), hidden, and bias
        proj4 = b + times(x, W) + times(dh, R)

        it_proj  = row_slice (proj4, 0*N, 1*N)  # split along stack_axis
        bit_proj = row_slice (proj4, 1*N, 2*N)
        ft_proj  = row_slice (proj4, 2*N, 3*N)
        ot_proj  = row_slice (proj4, 3*N, 4*N)

        it = sigmoid (it_proj)           # input gate(t)
        bit = it * activation (bit_proj) # applied to tanh of input network

        ft = sigmoid (ft_proj)           # forget-me-not gate(t)
        bft = ft * dc                    # applied to cell(t-1)

        ct = bft + bit                   # c(t) is sum of both

        ot = sigmoid (ot_proj)           # output gate(t)
        ht = ot * activation (ct)        # applied to tanh(cell(t))

        return (ht, ct)
    return lstm

def Embedding(N, name=''):
    E = Parameter((INFER,N), initializer=times_initializer)
    @Model(E=E)
    def embedding(x):
        return times(x,E)
    return embedding

def Sequential(functions):
    @Model(__items__=functions)
    def seq(x):
        for f in functions:
            x = f(x)
        return x
    return seq

#def Map(map_function):
#    @Model(map_function=map_function)
#    def map_f(x):
#        return map(map_function, x)
#    return map_f

def Fold(step_function, initial_state=0):
    # TODO: promote initial_state right away to a Constant() if it is a constant, to avoid repeated conversions. Same for Recurrence().
    initial_state = to_Variable(initial_state)
    @Model(step_function=step_function)
    def fold(x):
        s = initial_state  # state
        for sample in x:
            s = step_function(s, sample)
        return s[0] if isinstance(s, tuple) else s
    return fold

def Recurrence(step_function, initial_state=0):
    initial_state = to_Variable(initial_state)
    @Model(step_function=step_function)
    def recurrence(x):
        s = initial_state  # state
        out = []
        for sample in x:
            s = step_function(s, sample)
            out.append(s)
        return out
    return recurrence

# returns the set of Parameter objects hanging off a model
def get_parameters(m):
    parameters = set()
    for member_name in dir(m):
        if member_name[0] == '_' and member_name != '__items__':
            continue
        member = getattr(m, member_name)
        if isinstance(member, Parameter):
            parameters.add(member)
        elif member_name == '__items__':
            for item in member:
                parameters |= get_parameters(item)
        elif hasattr(member, '__ismodel__'):
            parameters |= get_parameters(member)
    return parameters

def dump_parameters(m, root='$'):
    for member_name in dir(m):
        if member_name[0] == '_' and member_name != '__items__':
            continue
        member = getattr(m, member_name)
        if isinstance(member, Parameter):
            print(root + '.' + member_name + ': ' + str(member.shape))
        elif member_name == '__items__':
            for i, item in enumerate(member):
                dump_parameters(item, '{}[{}]'.format(root, i))
        elif hasattr(member, '__ismodel__'):
            dump_parameters(member, root + '.' + member_name)

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

    # depth-first traversal
    #work_list = None # (node, tail)
    #visited = set() # [id(obj)] remembers every item that has ever been added to the work_list
    #def schedule(inputs):
    #    nonlocal work_list
    #    for input in reversed(inputs):
    #        if input not in visited:
    #            work_list = (input, work_list) # prepend the input
    #            visited.add(id(input))
    #schedule(roots)
    #while work_list:
    #    node, work_list = work_list # pop off first item
    #    # loop over its inputs
    #    schedule(node.inputs)
    #assert len(order) == len(visited)

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
    # ... can we just verify the basic invariant?
    seen = set()
    for node in order:
        for input in node.inputs:
            assert id(input) in seen # node must not be referenced as an input before it was seen
        seen.add(id(node))
    return order

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
    num_nodes = len(nodes)
    expected_num_ops = sum(1 for v in nodes if not v.computed)

    # management of batched operations
    ready_ops = dict()  # [key] -> list of Variables

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
    # scatter lazy and avoid it if possible)
    def transform_batched_op(op_batch):
        nonlocal num_compute_launches, num_gathers
        if len(op_batch) == 1: # short-circuit this to avoid unnecessary splice (...actually already taken care of the check for all inputs being the same)
            #v = op_batch[0]
            #v.compute_data()
            num_compute_launches += 1
            return
        # all ops are the same, so use the first as the reference
        v0 = op_batch[0]
        for inp in v0.inputs:
            assert isinstance(inp, Variable)
        is_mul = v0.op is cntk.NDArrayView.dot or v0.op is cntk.NDArrayView.dot_transpose
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
        num_inputs = len(v0.inputs)
        num_batched_ops = len(op_batch)
        def make_batched_input(i, inp_i_0): # returns either arg0 or a slice/splice
            args = tuple(v.inputs[i] for v in op_batch)
            arg0 = args[0]
            # matrix product is special, in that the right argument is always shared in the batch and not applied element-wise
            if is_mul and i == 1:
                assert all(arg is arg0 for arg in args)
                return arg0, False
            inp_i_0_shape = inp_i_0.shape if isinstance(inp_i_0, (Variable, np.ndarray)) else ()
            # check whether all inputs are the same (e.g. add a bias)--then don't batch
            sliced_from0 = getattr(arg0, 'sliced_from', None)
            def is_consecutive(i,arg):
                sliced_from = arg.sliced_from
                matchesd = sliced_from[0] is     sliced_from0[0]
                matchesi = sliced_from[1] == i + sliced_from0[1]
                return matchesd and matchesi
            if all(arg is arg0 for arg in args):
                # all the same args: use the object itself, assuming broadcasting
                return arg0, False
            elif arg0.op is Variable._op_alias and all((arg.op is Variable._op_alias and arg.inputs == arg0.inputs) for arg in args):
                # use the object itself, assuming broadcasting
                # This is the case where an identical op exists in all batch items, which got batched into a single one,
                # and the original reference was patched to an alias to that single one.
                return arg0, False
            #elif isinstance(arg0, Variable) and all(isinstance(arg, Variable) and (arg.data is arg0.data) for arg in args):
            #elif all((arg.data is arg0.data) for arg in args):
            #    # use the object itself, assuming broadcasting
            #    # TODO: What case is this? It must go away if we split off transformation.
            #    #   I think this is fully broadcast ops (same on all threads) that can be reduced back
            #    # TODO: problem if fully broadcast ops happen at different times; can that be? No--if one is ready, they are all ready at once
            #    # -> add a new mechanism like sliced_from--alias_of
            #    return arg0, False
            elif sliced_from0 and all(hasattr(arg, 'sliced_from') and is_consecutive(i, arg)
                                       for i, arg in enumerate(args)):
                # all inputs are consecutive views onto the same object--these came out of a previous batching op
                res = sliced_from0[0]
                # need to slice if range does not match, e.g. a sub-range or sub-index
                if res.shape[0] != num_batched_ops:
                    res = Variable((num_batched_ops,) + res.shape[1:],
                                   #lambda arg: arg[sliced_from0[1]:sliced_from0[1] + num_batched_ops],
                                   #[res])
                                   cntk.NDArrayView.__getitem__,
                                   [res],
                                   additional_args=(slice(sliced_from0[1], sliced_from0[1] + num_batched_ops),))
                return res, True
            else:
                # need to do actual splice
                nonlocal num_gathers
                num_gathers += 1
                return Variable((num_batched_ops,) + (1,) * (new_rank - ranks[i] - 1) + inp_i_0_shape,
                                lambda *args: cntk.NDArrayView.splice(*args, axis=ranks[i] - new_rank),
                                args), True
        batched_inputs_hasbatch = tuple(make_batched_input(i, inp_i_0)
                                        for i, inp_i_0 in enumerate(v0.inputs))
        batched_inputs = tuple(batched_input for batched_input, hasbatch in batched_inputs_hasbatch)
        hasbatch = any(tuple(hasbatch for batched_input, hasbatch in batched_inputs_hasbatch))
        #                    ^^ hasbatch is the same as argument == arg0; can simplify
        #for batched_input in batched_inputs: # and compute them
        #    if isinstance(batched_input, Variable) and not batched_input.computed: # (if already computed then no gather was necessary)
        #        batched_input.compute_data() # these are splice or splice--don't count either in num_compute_launches
                #print('out', batched_input.data.shape)
        # create a new node for the batched op
        # In some cases, neither input got batched. In that case, just execute a single op and distribute its output
        shape_batched = v0.shape
        if hasbatch:
            shape_batched = (num_batched_ops,) + shape_batched
        # if not hasbatch then all args are just v0's, so we can just compute v0 and broadcast it --split the code
        def to_batched_op(op):
            # if the operation is a reduction to (), we must modify it to not reduce over the batch axis
            # All ops in unary_reduction_ops are single-arg ops and are meant to accept an additional reduce_to_shape argument.
            if hasbatch and op in unary_reduction_ops:
                return lambda arg: op(arg, reduce_to_shape=(num_batched_ops,1)).reshape((num_batched_ops,))
            return op
        v_batched = Variable(shape_batched, to_batched_op(v0.op), batched_inputs, v0.backprop_to_functions, v0.additional_args)
        # if not hasbatch, we can just use v0 and replicate it over all others
        # now perform the operation batched
        #print(13, v_batched.op)
        #v_batched.compute_data()
        num_compute_launches += 1
        #print('out', v_batched.data.shape)
        # and copy the results back
        # We update the node in-place instead of replacing it and updating graph references.
        # That is because user code may also hold references that must remain valid.
        for i, v in enumerate(op_batch):
            if hasbatch:
                # mutate the op into a slice view into the new batched op
                v.sliced_from = (v_batched, i) # remember that this was sliced
                v.op = cntk.NDArrayView.__getitem__
                v.additional_args = (i,)
            else:
                # mutate the op into an alias into the new batched op (which is actually just a clone of v0)
                v.op = Variable._op_alias
                v.additional_args = ()
                # BUGBUG: commit 48b72d5b3925ca9661b3b975f6a4af13b2952648 broke batching of something, section after print() goes up from 8 to 121
            v.inputs = (v_batched,)
            #v.compute_data()
    # end of transform_batched_op

    # initialization
    #  - determine set of consumers for each node
    #  - set not-ready-inputs counter to #inputs
    #  - add any node with not-ready-children counter==0 to ready batched group
    for p in nodes:
        if p.computed:
            continue
        def make_key(p):
            # special case for matmul: right matrix must be identical to be batchable
            if p.op is cntk.NDArrayView.dot or p.op is cntk.NDArrayView.dot_transpose:
                return (p.op, ((p.inputs[0].shape, p.inputs[0].additional_args), (id(p.inputs[1]), p.inputs[1].additional_args)))
            # batch if both op and input shapes are the same
            return (p.op, tuple((v.shape, v.additional_args) for v in p.inputs))
        p.key = make_key(p)
        # TODO: must also include the storage format in the key; do this in C++ version
        p.consumers = []
        p.non_ready_inputs = 0
        for v in p.inputs:
            if isinstance(v, Variable) and not v.computed:
                v.consumers.append(p) # (due to topo sort, v is already initialized)
                p.non_ready_inputs += 1
        if p.non_ready_inputs == 0:
            add_ready(p)  # a leaf that's ready to go: make it part of the initial ready set

    # execute as long as still anything pending
    batches_run = 0
    ops_run = 0
    while ready_ops:
        # select the largest ready batch size
        key = max(ready_ops.keys(), key=(lambda key: len(ready_ops[key])))
        op_batch = ready_ops[key]
        # execute it
        #print('batch of', len(op_batch), 'for op', key)
        transform_batched_op(op_batch)
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

    # done
    print(ops_run, 'operations executed in', batches_run, 'batches, using an actual', num_compute_launches, 'compute launches and', num_gathers, 'gather launches')
    assert ops_run == expected_num_ops
    #for v in nodes:
    #    assert v.computed

# main evaluation function
#  - evaluate a set of root variables
#  - with full automatic dynamic batching
def batch_eval(vars):
    # transform the graph from raw (individual batch items) to the batched graph
    transform_to_batched_ops(vars)
    # BUGBUG: the following fails because we have a slice() somewhere which is not hashable
    #         It also seems to further change the graph, and results are not the same. So something is still wrong.
    #         Tested: This is not due to the topo_sort bug.
    #transform_to_batched_ops(vars)
    #transform_to_batched_ops(vars)
    # now actually compute the transformed graph
    nodes = topo_sort(vars)
    num_nodes = len(nodes)
    for p in nodes:
        if not p.computed:
            p.compute_data()
    #dump_graph(vars)

# gradient
# This computes the gradient of a variable (e.g. criterion) w.r.t. a set of model parameters, times an error signal.
# Call this only once for all parameters, otherwise you will duplicate computation (no caching!).
# Inputs:
#  - v: variable whose gradient is to be computed
#  - parameters: set of Parameter variables hanging off v to compute the gradient for
#  - error_signal: to back-propagate into the root
def create_gradient_graph(root, parameters, error_signal):
    # batch all ops
    # BUGBUG: This can only run once for now; so don't do it here
    #transform_to_batched_ops((root,))
    nodes = topo_sort([root])
    # determine set of nodes that depend on the parameters
    active_set = parameters.copy()
    for node in nodes:
        if any(input in active_set for input in node.inputs):
            active_set.add(node)
    if root not in active_set:
        # root not depending on any parameter (gradient is 0; we could just return 0 for those cases, but it's likely an error)
        raise ValueError('grad_times: variable does not depend on any of the given parameters')
    # now build the graph backwards
    # This is the error backpropagation algorithm in 12 lines of Python.
    gradients = dict() # [node] -> node's gradient; that is, error_signal * droot/dnode
    gradients[root] = error_signal
    for node in filter(lambda node: node in active_set, reversed(nodes)):
        g = gradients[node]
        # backprop into each child
        for i, input in enumerate(node.inputs):
            if input in active_set:
                input_g = node.backprop_to(i)(node, g)
                if input not in gradients:
                    gradients[input] = input_g
                else:
                    gradients[input] += input_g
    print(len(active_set), len(nodes), len(gradients))
    # gather the results
    res = { p: gradients[p] for p in parameters } # if a parameter does not feed root, then there will be no gradient, we will fail here
    # test computing the value of a gradient  --remove this later
    for p, g in res.items():
        g.get_value()
    return res

def dump_graph(vars): # vars can be a Variable or an iterable of Variables
    if isinstance(vars, Variable):
        vars = [vars]
    names = {} # [id(obj)] -> faked name
    def print_node(v):
        def name_it(v):
            if id(v) in names:
                return names[id(v)]
            name = "v" + str(len(names))
            try:
                name += '_' + v.op
            except:
                pass
            names[id(v)] = name
            return name
        def format_shape(v):
            if not isinstance(v, (np.ndarray, Variable)):
                return str(v) # e.g. a constant
            t = name_it(v) + ": "
            t += "Parameter" if isinstance(v, Parameter) else "Variable" if isinstance(v, Variable) else "ndarray"
            t += str(v.shape)
            if v.additional_args:
                t += ', ' + str(v.additional_args)
            return t
        t = format_shape(v)
        try:   # BUGBUG: why does it get here with an int?
            t += " = " + str(v.op) + "(" + ", ".join([format_shape(inp) for inp in v.inputs]) + ")"
        except AttributeError:
            pass
        print(t)
        pass
    order = topo_sort(vars)
    for node in order:
        print_node(node)
    return len(order)

from greenlet import greenlet # very lighweight coroutines

def train_minibatch(criterion, *batch_args):
    use_coroutines = True
    # for now, manually do the batch loop
    print('batch of', len(batch_args[0]))
    if use_coroutines:
        # create a coroutine for each batch entry
        # (a real implementation would switch to coroutines only upon encountering a need for the first time)
        zipped_batch_args = tuple(zip(*batch_args))
        coros = [lambda args=args: criterion(*args)[0] for args in zipped_batch_args] # (without the 'args=args', lambda will capture a reference to args, not its value; hence all see only its last value)
        # create the greenlet scheduler for the coroutines
        crits = [] # resulting values that we aggregate  --TODO: this is not general enough
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
                ce = coro()
                crits.append(ce)
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
                batch_eval(pending_vars) # for now
                pending_vars = []
            yield_to_next()
        prev_yield_to_batch_eval = Variable.set_yield(yield_to_batch_eval) # enable yielded batch computation
        yield_to_first()
        Variable.set_yield(prev_yield_to_batch_eval) # disable yielded batch computation
    else:
        crits = []
        for args in zip(*batch_args):
            ce, *_ = criterion(*args)
            crits.append(ce)
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
