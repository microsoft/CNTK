from dynamite import *

in_dim = 3  # 30000
embed_dim = 300
hidden_dim = 512
num_classes = 3 #0000

def create_model():
    encoder = Sequential([
        Embedding(embed_dim),
        #Recurrence(LSTM(hidden_dim), initial_state=(0,0)),
        Fold(LSTM(hidden_dim), initial_state=(0,0)),
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

# testing generators instead of functions

yield_sentinel = "yield_sentinel"

# sync version
def inner(m):
    return m + 1
def outer(n):
    y = inner(n)
    z = inner(42)
    return y + z

# async version
def inner_gen(m):
    #yield yield_sentinel  # cooperative task switch
    yield m + 1

def outer_gen(n):
    y = yield inner_gen(n)
    z = yield inner_gen(42)
    yield y + z

import inspect

# our coroutines can yield two values: A real one, and the yield_sentinel.
# Once they yield a real value, they are considered completed.
# run() returns that value.
# This scheduler itself has overhead; not clear how much. It must be implemented in C++ anyway.
def run(co):
    r = next(co)
    while True:
        if r is yield_sentinel: # cooperative task switched to happen here
            r = next(co)
            # note: a real scheduler would go round-robin here and maintain an explicit stack
        elif inspect.isgenerator(r): # we call another co-subroutine by returning its generator
            r = co.send(run(r)) # continue outer function with return value
        else:
            return r # done

def test_gen():
    # generator version
    start = timer()
    for i in range(10000):
        res = run(outer_gen(13))
    end = timer()
    assert res == 57
    print(end-start, "seconds for generator")
    # generator version
    start = timer()
    for i in range(10000):
        res = outer(13)
    end = timer()
    assert res == 57
    print(end-start, "seconds for direct call")
    # 1.6443511466666667 seconds for generator   --without actual yield_sentinel
    # 0.4049527466666667 seconds for direct call

if True:
    test_gen()

    p1 = Embedding(1)(1)
    xx = next(p1)
    v1 = plus(p1, 3 * np.array([[4]]))
    v2 = plus(p1, 5 * np.array([[6]]))
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
        #eval(crit)
    end = timer()
    dump_graph(crit)
    eval(crit)
    num_nodes = len(topo_sort(crit))
    num_samples = sum(len(batch_item[0]) for batch_item in mb.values())
    dur = (end - start) / repetitions
    print('dur:', dur, 'sec for', num_nodes, 'nodes, dur per node:', dur/num_nodes*1000,
          'ms, nodes/sec:', num_nodes/dur, 'samples/sec:', num_samples/dur)
