import numpy as np
import cntk as C
from cntk.layers.blocks import _INFERRED

def OptimizedRnnStack(hidden_dim, num_layers=1, recurrent_op='lstm', bidirectional=False, use_cudnn=True, name=''):
    if use_cudnn:
        W = C.parameter(_INFERRED + (hidden_dim,), init=C.glorot_uniform())
        def func(x):
            return C.optimized_rnnstack(x, W, hidden_dim, num_layers, bidirectional, recurrent_op=recurrent_op, name=name)
        return func
    else:
        def func(x):
            return C.splice(
                        C.layers.Recurrence(C.layers.LSTM(hidden_dim))(x),
                        C.layers.Recurrence(C.layers.LSTM(hidden_dim), go_backwards=True)(x),
                        name=name)
        return func

def HighwayBlock(dim, # ideally this should be inferred, but times does not allow inferred x inferred parameter for now
                 name=''):
    transform_weight_initializer=0
    transform_bias_initializer=0
    update_weight_initializer=0
    update_bias_initializer=0
    def func(x_var):
        x  = C.placeholder()
        WT = C.Parameter((dim,dim,), init=transform_weight_initializer, name=name+'_WT')
        bT = C.Parameter(dim,        init=transform_bias_initializer,   name=name+'_bT')
        WU = C.Parameter((dim,dim,), init=update_weight_initializer,    name=name+'_WU')
        bU = C.Parameter(dim,        init=update_bias_initializer,      name=name+'_bU')
        transform_gate = C.sigmoid(C.times(x, WT, name=name+'_T') + bT)
        update = C.relu(C.times(x, WU, name=name+'_U') + bU)
        return C.as_block(
            x + transform_gate * (update - x),
            [(x, x_var)],
            'HighwayBlock',
            'HighwayBlock'+name)
    return func
    
def HighwayNetwork(dim, highway_layers, name=''):
    return C.layers.For(range(highway_layers), lambda i : HighwayBlock(dim, name=name+str(i)))
    
def seq_loss(logits, y):
    prob = C.sequence.softmax(logits)
    return -C.log(C.sequence.last(C.sequence.gather(prob, y)))

def all_spans_loss(start_logits, start_y, end_logits, end_y):
    # this works as follows:
    # let end_logits be A, B, ..., Y, Z
    # let start_logits be a, b, ..., y, z
    # the tricky part is computing log sum (i<=j) exp(start_logits[i] + end_logits[j])
    # we break this problem as follows
    # x = logsumexp(A, B, ..., Y, Z), logsumexp(B, ..., Y, Z), ..., logsumexp(Y, Z), Z
    # y = a + logsumexp(A, B, ..., Y, Z), b + logsumexp(B, ..., Y, Z), ..., y + logsumexp(Y, Z), z + Z
    # now if we exponentiate each element in y we have all the terms we need. We just need to sum those exponentials...
    # logZ = last(sequence.logsumexp(y))
    x = C.layers.Recurrence(C.log_add_exp, go_backwards=True, initial_state=-1e+30)(end_logits)
    y = start_logits + x
    logZ = C.layers.Fold(C.log_add_exp, initial_state=-1e+30)(y)
    return logZ - C.sequence.last(C.sequence.gather(start_logits, start_y)) - C.sequence.last(C.sequence.gather(end_logits, end_y))

def seq_hardmax(logits):
    seq_max = C.layers.Fold(C.element_max, initial_state=C.constant(-1e+30, logits.shape))(logits)
    s = C.equal(logits, C.sequence.broadcast_as(seq_max, logits))
    s_acc = C.layers.Recurrence(C.plus)(s)
    return s * C.equal(s_acc, 1) # only pick the first one

class LambdaFunc(C.ops.functions.UserFunction):
    def __init__(self,
            arg,
            when=lambda arg: True,
            execute=lambda arg: print((len(arg), arg[0].shape,) if type(arg) == list else (1, arg.shape,), arg),
            name=''):
        self.when = when
        self.execute = execute

        super(LambdaFunc, self).__init__([arg], name=name)

    def infer_outputs(self):
        return [C.output_variable(self.inputs[0].shape, self.inputs[0].dtype, self.inputs[0].dynamic_axes)]

    def forward(self, argument, device=None, outputs_to_retain=None):
        if self.when(argument):
            self.execute(argument)

        return None, argument

    def backward(self, state, root_gradients):
        return root_gradients
        
    def clone(self, cloned_inputs):
        return self.__init__(*cloned_inputs)
        
def print_node(v):
    return C.user_function(LambdaFunc(v))