import numpy as np
import collections

INFER = 0
times_initializer="x" # for now a dummy string that is not None

class NDArray(np.ndarray):
    def __new__(cls, *args, **kwargs):
        return np.ndarray.__new__(cls, *args, **kwargs)

class Variable(NDArray):
    def __new__(cls, shape, op, inputs):
        #v = NDArray.__new__(cls, shape)
        v = NDArray.__new__(cls, tuple(0 for dim in shape))
        v.initializer = None
        v.op = op
        v.inputs = inputs
        return v

class Parameter(Variable):
    def __new__(cls,  shape, initializer=None):
        v = Variable.__new__(cls, shape, -1, [])
        v.initializer = None
        return v
    def __init__(self, shape, initializer=None):
        self.initializer = initializer
    
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
        v = Variable(a.shape if isinstance(a, np.ndarray) else (1,), opcode, (a,b))
        return v
    return f

def unary_op(opcode):
    def f(x):
        if isinstance(x, list):
            return map(f, x)
        v = Variable(x.shape, opcode, (x,))
        return v
    return f

@BroadcastingBinary
def times(a,b):
    if b.initializer is not None:
        b.resize((a.shape[0] if isinstance(a, np.ndarray) else 1, b.shape[1]), refcheck=False)
        b.initializer = None
    v = Variable((b.shape[1],), '@', (a,b))
    return v
plus = binary_op('+')
cross_entropy_with_softmax = binary_op('cross_entropy_with_softmax')
sigmoid = unary_op('sigmoid')
tanh = unary_op('tanh')
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

# TODO: this may be borked
def topo_sort(v):
    visited = set() # [id(obj)]
    stack = [v] if isinstance(v, np.ndarray) else []
    order = []
    while stack:
        v = stack.pop()
        if id(v) in visited:
            continue
        visited.add(id(v))
        if isinstance(v, Variable):
            for x in v.inputs:
                stack.insert(0,x) # (TODO: use bulk insertion)
        order.insert(0,v)
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
            if not isinstance(v, np.ndarray):
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
    #dump_graph(crit)
    num_nodes = len(topo_sort(crit))
    num_samples = sum(len(batch_item[0]) for batch_item in mb.values())
    dur = (end - start) / repetitions
    print('dur:', dur, 'sec for', num_nodes, 'nodes, dur per node:', dur/num_nodes*1000,
          'ms, nodes/sec:', num_nodes/dur, 'samples/sec:', num_samples/dur)
