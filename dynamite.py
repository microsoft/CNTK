import numpy as np
import collections

INFER = 0
times_initializer="x" # for now a dummy string that is not None

class Variable:
    def __new__(cls, shape, op, inputs):
        #v = NDArray.__new__(cls, tuple(0 for dim in shape))
        v = object.__new__(cls)
        v.shape = shape
        v.op = op
        v.inputs = inputs
        return v
    def __add__(self, other):
        return plus(self, other)

class Parameter(Variable):
    def __new__(cls,  shape, initializer=None):
        v = Variable.__new__(cls, shape, -1, [])
        return v
    def __init__(self, shape, initializer=None):
        if initializer:
            self.initializer = initializer
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
cross_entropy_with_softmax = binary_op('cross_entropy_with_softmax')

tanh = unary_op('tanh')
sigmoid = unary_op('sigmoid')
softmax = unary_op('softmax')

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
    W = Parameter((INFER,N))
    b = Parameter((N,))
    @Model(W=W, b=b)
    def dense(x):
        return activation(plus(times(x,W), b))
    return dense

def RNNBlock(N, activation=sigmoid):
    W = Parameter((INFER,N))
    R = Parameter((INFER,N))
    b = Parameter((N,))
    @Model(W=W, R=R, b=b)
    def rnn_block(s,x):
        return activation(plus(plus(times(s,R), times(x,W)), b))
    return rnn_block

def Embedding(N):
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

def Fold(step_function):
    @Model(step_function=step_function)
    def fold(x):
        s = 0  # state
        for sample in x:
            s = step_function(s, sample)
        return s
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

# ---

in_dim = 3  # 30000
embed_dim = 300
hidden_dim = 512
num_classes = 3 #0000

def create_model():
    encoder = Sequential([
        Embedding(embed_dim),
        Recurrence(RNNBlock(hidden_dim)),
        Fold(RNNBlock(hidden_dim)),
        Dense(num_classes)
    ])
    model = encoder
    return model

def read_minibatch():
    # returns list of arrays
    lens = range(10,35,1)   # a total input batch size of 550 time steps
    batch = {T:                             # item id
                ([13 for t in range(T)],    # input
                 42)                        # labels
             for T in lens}
    return batch

def train_minibatch(criterion, mb):
    # for now, manually do the batch loop
    crit = 0
    for inp, lab in mb.values():
        z = model(inp)
        ce = cross_entropy_with_softmax(z, lab)
        crit = plus(crit, ce)
    return crit

if True:
    from timeit import default_timer as timer

    p1 = Embedding(1)(1)
    v1 = plus(p1, times(3, np.array([[4]])))
    v2 = plus(p1, times(5, np.array([[6]])))
    v = v1 + v2
    dump_graph(v)

    model = create_model()
    print(dir(model))
    #model[0]
    dump_parameters(model)
    def criterion(inp, lab):
        z = model(inp)
        ce = cross_entropy_with_softmax(z, lab)
        return ce
    mb = read_minibatch()  # (input: list[Sequence[tensor]]], labels: list[tensor]])
    start = timer()
    repetitions = 10
    for count in range(repetitions):
        crit = train_minibatch(criterion, mb)
    end = timer()
    dump_graph(crit)
    num_nodes = len(topo_sort(crit))
    num_samples = sum(len(batch_item[0]) for batch_item in mb.values())
    dur = (end - start) / repetitions
    print('dur:', dur, 'sec for', num_nodes, 'nodes, dur per node:', dur/num_nodes*1000,
          'ms, nodes/sec:', num_nodes/dur, 'samples/sec:', num_samples/dur)
