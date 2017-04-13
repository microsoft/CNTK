from .variable import *

##############################################################################
#
# Dynamite layers library
#
##############################################################################

# TODO: move to contrib/dynamite/layers.py ; import .variable

# Layers are function objects that contain parameters and form a hierarchy that
# can be traversed in order to identity model parameters, e.g. for training or persisting.
# The layers library here is a functional wrapper over the Variable class.
# The Variable class knows nothing about the layers.

# --- decorators

# decorators for layers that accept tensors as well as collections, such as Embedding() and Dense()
# TODO: this map() operation should be batch_map, in case the function passed in yields. We'd need a nested scheduler for that.
def BroadcastingUnary(unary_op):
    # BUGBUG: testing for nested sequences must know how to align... how to do that?
    def broadcasting_unary_op(x):
        if isinstance(x, (list, tuple)): # broadcast along sequence
            return map(broadcasting_unary_op, x) # recursive, to support collections of collections  --TODO: batch_map()
        return unary_op(x)
    return broadcasting_unary_op

#def BroadcastingBinary(binary_op):
#    # BUGBUG: testing for nested sequences must know how to align... how to do that?
#    def broadcasting_binary_op(a,b):
#        if isinstance(a, list):
#            if isinstance(b, list):
#                return [broadcasting_binary_op(x,y) for x,y in zip(a,b)]
#            return map(lambda sample: broadcasting_binary_op(sample, b), a)
#        elif isinstance(b, list):
#            return map(lambda sample: broadcasting_binary_op(a, sample), b)
#        return binary_op(a,b)
#    return broadcasting_binary_op

# --- layers

def identity(x):
    return x

def Dense(N, bias=True, activation=identity, name=''):
    W = Parameter((INFER,N), initializer=times_initializer)
    if bias:
        b = Parameter((N,))
        @Model(W=W, b=b)
        @BroadcastingUnary
        def dense(x):
            return activation(x @ W + b)
    else:
        @Model(W=W)
        @BroadcastingUnary
        def dense(x):
            return activation(x @ W)
    return dense

def LogValues(): # fake layer to print the value as it passes through; for testing direct-mode values/co-routines
    @BroadcastingUnary
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
        return activation(x @ W + b + s @ R)  # this ordering allows the first two to be batched
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
    @BroadcastingUnary
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

def Map(map_function): # TODO: this has never been tested
    @Model(map_function=map_function)
    @BroadcastingUnary # this realizes the (nested) map
    def map_f(x):
        return map_function(x)
    return BroadcastingUnary(f)

def Fold(step_function, initial_state=0):
    # TODO: promote initial_state right away to a Constant() if it is a constant, to avoid repeated conversions. Same for Recurrence().
    #initial_state = to_Variable(initial_state)  # HACK
    @Model(step_function=step_function)
    def fold(x):
        s = initial_state  # state
        for sample in x:
            s = step_function(s, sample)
        return s[0] if isinstance(s, tuple) else s
    return fold

def Recurrence(step_function, initial_state=0):
    #initial_state = to_Variable(initial_state)
    @Model(step_function=step_function)
    def recurrence(x):
        s = initial_state  # state
        out = []
        for sample in x:
            s = step_function(s, sample)
            out.append(s)
        return out
    return recurrence

def Model(**kwargs):
    def patch(f):
        f.__ismodel__ = True
        f.get_parameters      = lambda: _Model_get_parameters(f)
        f.get_parameter_names = lambda: _Model_get_parameter_names(f)
        for name, value in kwargs.items():
            setattr(f, name, value) # add all as class members
        #def mygetitem(self, x):
        #    x
        #f.__getitem__ = mygetitem
        return f
    return patch

# --- Model decorator

# returns the set of Parameter objects hanging off a model
# Returned as a tuple, since sets in Python have non-deterministic ordering.
def _Model_get_parameters(m):
    parameters = []
    for member_name in dir(m):
        if member_name[0] == '_' and member_name != '__items__':
            continue
        member = getattr(m, member_name)
        if isinstance(member, Parameter):
            parameters.append(member)
        elif member_name == '__items__':
            for item in member:
                parameters.extend(_Model_get_parameters(item))
        elif hasattr(member, '__ismodel__'):
            parameters.extend(_Model_get_parameters(member))
    return tuple(parameters)

# returns a dict [parameter] -> its name in the hierarchy
def _Model_get_parameter_names(m, root='_'):
    parameter_names = dict()
    for member_name in dir(m):
        if member_name[0] == '_' and member_name != '__items__':
            continue
        member = getattr(m, member_name)
        if isinstance(member, Parameter):
            parameter_names[member] = root + '.' + member_name
        elif member_name == '__items__':
            for i, item in enumerate(member):
                parameter_names.update(_Model_get_parameter_names(item, '{}[{}]'.format(root, i)))
        elif hasattr(member, '__ismodel__'):
            parameter_names.update(_Model_get_parameter_names(member, root + '.' + member_name))
    return parameter_names

# (delete this if no longer needed)
def delete_this_dump_parameters(m, root='$'):
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
