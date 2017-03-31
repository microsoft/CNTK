# Variable: deferred computation
# Value: GPU direct object
#
# >>> d = gpu(0)
# >>> x=internal.sanitize_value((3,), a, np.float32, gpu(0))    # returns an NDArrayView
# >>> x
# <cntk.core.NDArrayView; proxy of <Swig Object of type 'CNTK::NDArrayViewPtr *' at 0x0000003ABB2EDBD0> >
# >>> x.to_ndarray()
# array([  1.,  15.,   3.], dtype=float32)

import numpy as np
import collections
from timeit import default_timer as timer

INFER = 0
times_initializer="x" # for now a dummy string that is not None

class Variable:
    def __new__(cls, shape, op, inputs):
        #v = NDArray.__new__(cls, tuple(0 for dim in shape))
        v = object.__new__(cls)
        v.shape = shape
        v.op = op
        v.inputs = inputs
        v.computed = False
        return v
    def __add__(self, other):
        return plus(self, other)
    def __mul__(self, other):
        return element_times(self, other)
    def __matmul__(self, other):
        return times(self, other)

class Parameter(Variable):
    def __new__(cls,  shape, initializer=None):
        v = Variable.__new__(cls, shape, 'Parameter', [])
        return v
    def __init__(self, shape, initializer=None):
        if initializer:
            self.initializer = initializer
        self.computed = True
    def resize(self, shape, refcheck=False):
        self.shape = shape
    
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

def binary_op(opcode):
    @BroadcastingBinary
    def f(a,b):
        return Variable((max(a.shape[0] if isinstance(a, (np.ndarray, Variable)) else 1, # broadcasting
                             b.shape[0] if isinstance(b, (np.ndarray, Variable)) else 1
                             ),), opcode, (a,b))
    return f

def reducing_binary_op(opcode):
    @BroadcastingBinary
    def f(a,b):
        return Variable((1,), opcode, (a,b))
    return f

def unary_op(opcode):
    def f(x):
        if isinstance(x, list): # broadcast along sequence
            return map(f, x)
        return Variable(x.shape, opcode, (x,))
    return f

@BroadcastingBinary
def times(a,b):
    if hasattr(b, 'initializer'):
        b.resize((a.shape[0] if isinstance(a, (np.ndarray, Variable)) else 1,
                  b.shape[1]),
                 refcheck=False)
        del b.initializer
    return Variable((b.shape[1],), '@', (a,b))

plus = binary_op('+')
element_times = binary_op('*')
cross_entropy_with_softmax = reducing_binary_op('cross_entropy_with_softmax')
classification_error = reducing_binary_op('classification_error')

tanh = unary_op('tanh')
sigmoid = unary_op('sigmoid')
softmax = unary_op('softmax')
row_slice_0 = unary_op('row_slice')
def row_slice(x, begin, end):
    return row_slice_0(x) # (ignore dims for this test)

def identity(x):
    return x

def Model(**kwargs):
    def patch(f):
        f.__ismodel__ = True
        for name, value in kwargs.items():
            setattr(f, name, value) # add all as class members
        #def mygetitem(self, x):
        #    x
        #f.__getitem__ = mygetitem
        return f
    return patch

def Dense(N, activation=identity):
    W = Parameter((INFER,N), initializer=times_initializer)
    b = Parameter((N,))
    @Model(W=W, b=b)
    def dense(x):
        return activation(x @ W + b)
    return dense

def RNNBlock(N, activation=sigmoid):
    W = Parameter((INFER,N), initializer=times_initializer)
    R = Parameter((INFER,N), initializer=times_initializer)
    b = Parameter((N,))
    @Model(W=W, R=R, b=b)
    def rnn_block(s,x):
        return activation(plus(plus(times(s,R), times(x,W)), b))
    return rnn_block

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

def Embedding(N):
    E = Parameter((INFER,N), initializer=times_initializer)
    @Model(E=E)
    def embedding(x):
        yield times(x,E)
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
    @Model(step_function=step_function)
    def fold(x):
        s = initial_state  # state
        for sample in x:
            s = step_function(s, sample)
        return s[0] if isinstance(s, tuple) else s
    return fold

def Recurrence(step_function):
    @Model(step_function=step_function)
    def recurrence(x):
        s = 0  # state
        out = []
        for sample in x:
            s = step_function(s, sample)
            out.append(s)
        return out
    return recurrence

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
def topo_sort(v):
    if not isinstance(v, (np.ndarray, Variable)):
       return
    visited = set() # [id(obj)]
    stack = []
    order = []
    stack.append(v)
    visited.add(id(v))
    num_implanted = 0
    while stack:
        p = stack.pop()
        for v in p.inputs:
            if isinstance(v, Variable):
                if id(v) in visited:
                    continue
                if p:
                    v.parent = p # once we emit the first one, we can emit its parent, too
                    num_implanted += 1 # (sanity check only)
                    p = None
                stack.append(v)
                visited.add(id(v))
        while p:  # no children (left) to process -> we can emit this and all parents that are ready
            order.append(p)
            q = getattr(p, 'parent', None)
            if q:
                del p.parent # clean up after ourselves (may not be needed)
                num_implanted -= 1 # (sanity check only)
            p = q
    assert num_implanted == 0
    assert len(order) == len(visited)
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
def eval(v):
    if not isinstance(v, Variable):
       return v
    nodes = topo_sort(v)    # (it is possible to implement this without, just more complex)
    num_nodes = len(nodes)
    expected_num_ops = sum(1 for v in nodes if not v.computed)

    # management of batched operations
    def shape_of(v):
        if isinstance(v, (np.ndarray, Variable)):
            return v.shape
        else:
            return ()

    ready_ops = dict()  # [key] -> list of Variables

    def add_ready(v):
        key = v.key
        if key not in ready_ops:
            ready_ops[key] = [v] # first entry: create
        else:
            ready_ops[key].append(v)

    # initialization
    #  - determine set of consumers for each node
    #  - set not-ready-inputs counter to #inputs
    #  - add any node with not-ready-children counter==0 to ready batched group
    for p in nodes:
        if p.computed:
            continue
        p.key = (p.op, tuple(shape_of(v) for v in p.inputs)) # batch if both op and input shapes are the same
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
        batches_run += 1
        # done with this one
        #  - for each member of the batched op, check each consumer whether it is now ready; if so, move to ready set
        #  - delete the batched group
        del ready_ops[key]  # remove from set before we add the newly ready ones
        for v in op_batch:
            assert not v.computed
            v.computed = True # value is available now
            ops_run += 1
            for p in v.consumers:
                assert p.non_ready_inputs > 0
                p.non_ready_inputs -= 1
                if p.non_ready_inputs == 0:
                    add_ready(p)

    # done
    #print(ops_run, 'operations executed in', batches_run, 'batches')
    #assert ops_run == expected_num_ops
    #for v in nodes:
    #    assert v.computed


def dump_graph(v):
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
            return t
        t = format_shape(v)
        try:   # BUGBUG: why does it get here with an int?
            t += " = " + str(v.op) + "(" + ", ".join([format_shape(inp) for inp in v.inputs]) + ")"
        except AttributeError:
            pass
        print(t)
        pass
    order = topo_sort(v)
    for node in order:
        print_node(node)
    return len(order)
