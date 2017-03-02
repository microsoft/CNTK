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

#def HighwayBlock()


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