import cntk as C


def PastValueWindow(window_size, axis, name=''):
    '''
    Layer factory function to create a function that returns a static, maskable view for N past steps over a sequence along the given 'axis'.
    It returns two matrices: a value matrix, shape=(N,dim), and a valid window, shape=(1,dim)
    '''

    # helper to get the nth element
    def nth(input, offset):
        return C.sequence.last(C.past_value(input, time_step=offset+1))

    def past_value_window(x):

        ones_like_input = C.sequence.broadcast_as(1, x)

        # get the respective n-th element from the end
        last_values = [nth(x, t) for t in range(window_size)]
        last_valids = [nth(ones_like_input, t) for t in range(window_size)]

        # stack rows 'beside' each other in a new static axis (create a new static axis that doesn't exist)
        value = C.splice(*last_values, axis=axis, name='value')
        valid = C.splice(*last_valids, axis=axis, name='valid')

        # value[t] = value of t steps back; valid[t] = true if there was a value t steps back
        return value, valid

    return past_value_window

def HighwayBlock(dim, # ideally this should be inferred, but times does not allow inferred x inferred parameter for now
                 transform_weight_initializer=C.normal(scale=1),
                 transform_bias_initializer=0,
                 update_weight_initializer=C.normal(scale=1),
                 update_bias_initializer=0,
                 name=''):
    x  = C.placeholder_variable()
    WT = C.Parameter(C.blocks._INFERRED + (dim,), init=transform_weight_initializer, name=name+'_WT')
    bT = C.Parameter(dim,                         init=transform_bias_initializer,   name=name+'_bT')
    WU = C.Parameter(C.blocks._INFERRED + (dim,), init=update_weight_initializer,    name=name+'_WU')
    bU = C.Parameter(dim,                         init=update_bias_initializer,      name=name+'_bU')
    transform_gate = C.sigmoid(C.times(x, WT, name=name+'_T') + bT)
    update = C.relu(C.times(x, WU, name=name+'_U') + bU)
    return x + transform_gate * (update - x)
    
def HighwayNetwork(dim, highway_layers, name=''):
    return C.For(range(highway_layers), lambda i : HighwayBlock(dim, name=name+str(i)))

def seqlogZ(seq):
    x = C.placeholder_variable(shape=seq.shape, dynamic_axes=seq.dynamic_axes)
    #print('x',x)
    logaddexp = C.log_add_exp(seq, C.past_value(x, initial_state=C.constant(-1e+30,seq.shape)))
    #print('lse',logaddexp)
    logaddexp.replace_placeholders({x: logaddexp})
    #print('lse++', logaddexp)
    ret = C.sequence.last(logaddexp)
    #print('ret', ret)
    return ret

class LambdaFunc(C.ops.functions.UserFunction):
    def __init__(self,
            arg,
            when=lambda arg: True,
            execute=lambda arg: print(arg),
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