import cntk as C
import numpy as np
from cntk import layers, blocks, models
from helpers import PastValueWindow, seqlogZ
import pickle

with open('vocabs.pkl', 'rb') as vf:
    known, vocab, chars = pickle.load(vf)

word_size = 20
w_dim = len(vocab)
c_dim = len(chars)*word_size
a_dim = 1
dim = 100
convs = 100
rf = 5

C.set_default_device(C.cpu())

# todo are the char cnns sharing parameters?
def charcnn(x):
    return C.models.Sequential([
        C.layers.Convolution1D((5,), convs, activation=C.relu, init=C.glorot_uniform(), pad=[True], strides=1, bias=True, init_bias=True),
        C.GlobalMaxPooling()])(x)

# todo are the lstms sharing parameters?
def bilstm(x):
    return C.splice(C.Recurrence(C.blocks.LSTM(dim))(x), C.Recurrence(C.blocks.LSTM(dim),go_backwards=True)(x))

def combine_glove_and_learnable(known, vocab):
    # load glove
    npglove = np.zeros((known,dim), dtype=np.float32)
    with open('glove.6B.100d.txt', encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            word = parts[0].lower()
            if word in vocab:
                npglove[vocab[word],:] = np.asarray([float(p) for p in parts[1:]])
    glove = C.constant(npglove)
    nonglove = C.parameter(shape=(len(vocab) - known, dim), init=C.glorot_uniform())
    return C.splice(glove, nonglove, axis=0)

c = C.Axis.new_unique_dynamic_axis('c')
q = C.Axis.new_unique_dynamic_axis('q')
b = C.Axis.default_batch_axis()
cw = C.input_variable(w_dim, dynamic_axes=[b,c], is_sparse=True, name='cw')
qw = C.input_variable(w_dim, dynamic_axes=[b,q], is_sparse=True, name='qw')
cc = C.input_variable(c_dim, dynamic_axes=[b,c], name='cc')
qc = C.input_variable(c_dim, dynamic_axes=[b,q], name='qc')
ab = C.input_variable(a_dim, dynamic_axes=[b,c], name='ab')
ae = C.input_variable(a_dim, dynamic_axes=[b,c], name='ae')

mb_source = C.MinibatchSource(C.CTFDeserializer('val.ctf', C.StreamDefs(
    context_words = C.StreamDef('cw',  shape=w_dim, is_sparse=True),
    query_words   = C.StreamDef('qw',  shape=w_dim, is_sparse=True),
    answer_begin  = C.StreamDef('ab',  shape=1,     is_sparse=False),
    answer_end    = C.StreamDef('ae',  shape=1,     is_sparse=False),
    context_chars = C.StreamDef('cc',  shape=c_dim, is_sparse=True),
    query_chars   = C.StreamDef('qc',  shape=c_dim, is_sparse=True)
)), randomize=False)

input_map = {
    cw: mb_source.streams.context_words,
    qw: mb_source.streams.query_words,
    cc: mb_source.streams.context_chars,
    qc: mb_source.streams.query_chars,
    ab: mb_source.streams.answer_begin,
    ae: mb_source.streams.answer_end
}

mb_data = mb_source.next_minibatch(256, input_map=input_map)
print(mb_data)

embedding = combine_glove_and_learnable(known, vocab)

qchars = C.reshape(qc,(qc.shape[0]//20,20))

# todo we need to reshape because GlobalMaxPooling is retaining a trailing singleton dimension
q_spliced_input = C.splice(C.times(qw, embedding), C.reshape(charcnn(qchars), convs))

# todo highway network

# todo replace -2 with new_axis() once new_axis() is working properly
pvw_input = bilstm(q_spliced_input)
pvw = PastValueWindow(15, -2)(pvw_input)

print(pvw[0])

cchars = C.reshape(cc,(cc.shape[0]//20,20))

# todo we need to reshape because GlobalMaxPooling is retaining a trailing singleton dimension
c_spliced_input = C.splice(C.times(cw, embedding), C.reshape(charcnn(cchars), convs))

# todo highway network
c_lstm_output = bilstm(c_spliced_input)
print('lstm output',c_lstm_output)

# This part deserves some explanation
# It is the attention layer
# In the paper they use a 6 * dim dimensional vector
# here we split it in three parts because the different parts
# participate in very different operations
# so W * [h; u; h.* u] becomes w1 * h + w2 * u + w3 * (h.*u)
ws1 = C.parameter(shape=(2 * dim, 1), init=C.glorot_uniform())
ws2 = C.parameter(shape=(2 * dim, 1), init=C.glorot_uniform())
ws3 = C.parameter(shape=(1, 2 * dim), init=C.glorot_uniform())

wh = C.times (c_lstm_output, ws1)
wu = C.reshape(C.times (pvw[0], ws2), (15,))
whu = C.times_transpose(c_lstm_output, C.sequence.broadcast_as(C.element_times (pvw[0], ws3), c_lstm_output))
S = wh + whu + C.sequence.broadcast_as(wu, c_lstm_output)
q_logZ = C.reshape(C.reduce_log_sum_exp(S),(1,))
q_attn = C.reshape(C.exp(S - q_logZ),(-1,1))
utilde = C.reshape(C.reduce_sum(C.sequence.broadcast_as(pvw[0], q_attn) * q_attn, axis=0),(-1))
print(utilde)

max_col = C.reduce_max(S)
c_logZ = seqlogZ(max_col)
c_attn = C.exp(max_col - C.sequence.broadcast_as(c_logZ, max_col))
htilde = C.sequence.reduce_sum(c_lstm_output * c_attn)
print(htilde)
print(c_lstm_output)

Htilde = C.sequence.broadcast_as(htilde, c_lstm_output)
modeling_layer_input = C.splice(c_lstm_output, utilde, c_lstm_output * utilde,  c_lstm_output * Htilde)
print(modeling_layer_input)

first_layer = bilstm(modeling_layer_input)
second_layer = bilstm(first_layer)
third_layer = bilstm(second_layer)

def seqloss(logits, y):
    return C.sequence.last(C.sequence.gather(logits, y)) - seqlogZ(logits)

begin_input = C.splice(modeling_layer_input, second_layer)
begin_weights = C.parameter(shape=(C.InferredDimension,1), init=C.glorot_uniform())
begin_logits = C.times(begin_input, begin_weights)
begin_loss = seqloss(begin_logits, ab)

end_input = C.splice(modeling_layer_input, third_layer)
end_weights = C.parameter(shape=(C.InferredDimension,1), init=C.glorot_uniform())
end_logits = C.times(end_input, end_weights)
end_loss = seqloss(end_logits, ae)

loss = begin_loss + end_loss

#print(loss.grad(mb_data))
func = loss
print([t.shape for t in func.grad(mb_data, wrt=loss.parameters)])