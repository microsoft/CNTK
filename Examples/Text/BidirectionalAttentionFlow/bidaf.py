import cntk as C
import numpy as np
from cntk import layers, blocks, models
from helpers import *
from config import *
import pickle

with open('vocabs.pkl', 'rb') as vf:
    known, vocab, chars = pickle.load(vf)

word_count_threshold = data_config['word_count_threshold']
char_count_threshold = data_config['char_count_threshold']
word_size = data_config['word_size']
max_query_len = data_config['max_query_len']
max_context_len = data_config['max_context_len']

wg_dim = known
wn_dim = len(vocab) - known
c_dim = len(chars) * word_size
a_dim = 1

dim = training_config['hidden_dim']
convs = training_config['char_convs']
dropout = training_config['dropout']
char_emb_dim = training_config['char_emb_dim']
highway_layers = training_config['highway_layers']

#C.set_default_device(C.cpu())
class PolyMath:
    @staticmethod
    def charcnn(x):
        conv_out = C.models.Sequential([
            C.layers.Embedding(char_emb_dim),
            C.layers.Dropout(0.2),
            C.layers.Convolution1D((5,), convs, activation=C.relu, init=C.glorot_uniform(), pad=[True], strides=1, bias=True, init_bias=True)])(x)
        return C.reduce_max(conv_out, axis=1) # workaround cudnn failure in GlobalMaxPooling

    @staticmethod
    def embed(known, vocab):
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
        
        def func(wg, wn):
            return wg @ glove + wn @ nonglove
        return func

    @staticmethod
    def model():
        c = C.Axis.new_unique_dynamic_axis('c')
        q = C.Axis.new_unique_dynamic_axis('q')
        b = C.Axis.default_batch_axis()
        cgw = C.input_variable(wg_dim, dynamic_axes=[b,c], is_sparse=True, name='cgw')
        cnw = C.input_variable(wn_dim, dynamic_axes=[b,c], is_sparse=True, name='cnw')
        qgw = C.input_variable(wg_dim, dynamic_axes=[b,q], is_sparse=True, name='qgw')
        qnw = C.input_variable(wn_dim, dynamic_axes=[b,q], is_sparse=True, name='qnw')
        cc = C.input_variable(c_dim, dynamic_axes=[b,c], name='cc')
        qc = C.input_variable(c_dim, dynamic_axes=[b,q], name='qc')
        ab = C.input_variable(a_dim, dynamic_axes=[b,c], name='ab')
        ae = C.input_variable(a_dim, dynamic_axes=[b,c], name='ae')

        embedding = PolyMath.embed(wg_dim, vocab)

        input_chars = C.placeholder_variable(shape=(c_dim,))
        input_glove_words = C.placeholder_variable(shape=(wg_dim,))
        input_nonglove_words = C.placeholder_variable(shape=(wn_dim,))

        # we need to reshape because GlobalMaxPooling/reduce_max is retaining a trailing singleton dimension
        # todo GlobalPooling/reduce_max should have a keepdims default to False
        embedded = C.splice(embedding(input_glove_words, input_nonglove_words), C.reshape(PolyMath.charcnn(C.reshape(input_chars, (C.InferredDimension, word_size))), convs))

        input_layers = C.layers.Sequential([
            HighwayNetwork(dim=(embedded.shape[0]), highway_layers=2),
            C.Dropout(0.2),
            OptimizedRnnStack(dim, bidirectional=True),
        ])
        
        q_emb = embedded.clone(C.CloneMethod.share, dict(zip(embedded.placeholders, [qgw,qnw,qc])))
        c_emb = embedded.clone(C.CloneMethod.share, dict(zip(embedded.placeholders, [cgw,cnw,cc])))

        q_processed = input_layers(q_emb) # synth_embedding for query
        c_processed = input_layers(c_emb) # synth_embedding for context

        #convert query's sequence axis to static
        qvw, qvw_mask = ValueWindow(max_query_len, C.Axis.new_leading_axis())(q_processed)
        print('qvw', qvw)

        #qvw_c = C.reduce_sum(qvw, 1)
        #qvw_c = print_node(qvw_c)
        #qvw = C.times(qvw_c, np.ones((1,200))) + qvw

        # This part deserves some explanation
        # It is the attention layer
        # In the paper they use a 6 * dim dimensional vector
        # here we split it in three parts because the different parts
        # participate in very different operations
        # so W * [h; u; h.* u] becomes w1 * h + w2 * u + w3 * (h.*u)
        ws1 = C.parameter(shape=(2 * dim, 1), init=C.glorot_uniform())
        ws2 = C.parameter(shape=(2 * dim, 1), init=C.glorot_uniform())
        ws3 = C.parameter(shape=(1, 2 * dim), init=C.glorot_uniform())
        att_bias = C.parameter(shape=(max_query_len,), init=0)

        wh = C.times (c_processed, ws1)
        wu = C.reshape(C.times (qvw, ws2), (max_query_len,))
        whu = C.times_transpose(c_processed, C.sequence.broadcast_as(C.element_times (qvw, ws3), c_processed))
        S = wh + whu + C.sequence.broadcast_as(wu, c_processed) + att_bias
        # mask out values outside of Query, and fill in gaps with -1e+30 as neutral value for both reduce_log_sum_exp and reduce_max
        qvw_mask_expanded = C.sequence.broadcast_as(C.reshape(qvw_mask, (max_query_len,)), c_processed)
        S = C.element_select(qvw_mask_expanded, S, C.constant(-1e+30, S.shape))
        q_logZ = C.reshape(C.reduce_log_sum_exp(S),(1,))
        q_attn = C.reshape(C.exp(S - q_logZ),(-1,1))
        #q_attn = print_node(q_attn)
        utilde = C.reshape(C.reduce_sum(C.sequence.broadcast_as(qvw, q_attn) * q_attn, axis=0),(-1))
        print('utilde', utilde)

        max_col = C.reduce_max(S)
        c_logZ = C.layers.Fold(C.log_add_exp, initial_state=C.constant(-1e+30, max_col.shape))(max_col)
        c_attn = C.exp(max_col - C.sequence.broadcast_as(c_logZ, max_col))
        htilde = C.sequence.reduce_sum(c_processed * c_attn)
        print('htilde', htilde)
        print('c_processed', c_processed)

        Htilde = C.sequence.broadcast_as(htilde, c_processed)
        att_context = C.splice(c_processed, utilde, c_processed * utilde,  c_processed * Htilde)
        print('att_context', att_context)

        #modeling layer
        #todo replace with optimized_rnnstack for training purposes once it supports dropout
        modeling_layer = C.layers.Sequential([
            C.Dropout(0.2),
            OptimizedRnnStack(dim, bidirectional=True),
            C.Dropout(0.2),
            OptimizedRnnStack(dim, bidirectional=True)
        ])

        mod_context = modeling_layer(att_context)
        print(mod_context)

        #output layer
        start_logits = C.layers.Dense(1)(C.splice(mod_context, att_context))
        
        start_hardmax = seq_hardmax(start_logits)
        att_mod_ctx = C.sequence.last(C.sequence.gather(mod_context, start_hardmax))
        att_mod_ctx_expanded = C.sequence.broadcast_as(att_mod_ctx, att_context)
        end_input = C.splice(att_context, mod_context, att_mod_ctx_expanded, mod_context * att_mod_ctx_expanded)
        m2 = OptimizedRnnStack(dim, bidirectional=True)(end_input)
        end_logits = C.layers.Dense(1)(m2)

        start_loss = seq_loss(start_logits, ab)
        end_loss = seq_loss(end_logits, ae)
        end_hardmax = seq_hardmax(end_logits)

        loss = start_loss + end_loss
        
        return C.combine([start_hardmax, end_hardmax]), loss

    @staticmethod    
    def f1_score(ab, ae, oab, oae):
        answers = C.splice(ab, ae, oab, oae)
        avw, _ = ValueWindow(max_context_len, C.Axis.new_leading_axis())(answers)
        answer_indices = C.argmax(avw, 0)
        answer_indices = print_node(answer_indices)
        gt_start = C.slice(answer_indices, 1, 0, 1)
        gt_end = C.slice(answer_indices, 1, 1, 2)
        test_id1 = C.slice(answer_indices, 1, 2, 3)
        test_id2 = C.slice(answer_indices, 1, 3, 4)
        test_start = C.element_min(test_id1, test_id2)
        test_end = C.element_max(test_id1, test_id2)
        common_start = C.element_max(test_start, gt_start)
        common_end = C.element_max(C.element_min(test_end, gt_end), common_start - 1)
        common_len = common_end - common_start + 1
        test_len = test_end - test_start + 1
        gt_len = gt_end - gt_start + 1
        precision = common_len / test_len
        recall = common_len / gt_len
        return precision * recall * 2 / (precision + recall)

def training():
    z, loss = PolyMath.model()

    mb_source = C.MinibatchSource(C.CTFDeserializer('val.ctf', C.StreamDefs(
        context_g_words  = C.StreamDef('cgw', shape=wg_dim, is_sparse=True),
        query_g_words    = C.StreamDef('qgw', shape=wg_dim, is_sparse=True),
        context_ng_words = C.StreamDef('cnw', shape=wn_dim, is_sparse=True),
        query_ng_words   = C.StreamDef('qnw', shape=wn_dim, is_sparse=True),
        answer_begin     = C.StreamDef('ab',  shape=1,      is_sparse=False),
        answer_end       = C.StreamDef('ae',  shape=1,      is_sparse=False),
        context_chars    = C.StreamDef('cc',  shape=c_dim,  is_sparse=True),
        query_chars      = C.StreamDef('qc',  shape=c_dim,  is_sparse=True)
    )), randomize=False)

    input_map = {
        argument_by_name(loss, 'cgw'): mb_source.streams.context_g_words,
        argument_by_name(loss, 'qgw'): mb_source.streams.query_g_words,
        argument_by_name(loss, 'cnw'): mb_source.streams.context_ng_words,
        argument_by_name(loss, 'qnw'): mb_source.streams.query_ng_words,
        argument_by_name(loss, 'cc' ): mb_source.streams.context_chars,
        argument_by_name(loss, 'qc' ): mb_source.streams.query_chars,
        argument_by_name(loss, 'ab' ): mb_source.streams.answer_begin,
        argument_by_name(loss, 'ae' ): mb_source.streams.answer_end
    }

    err = 1 - PolyMath.f1_score(argument_by_name(loss, 'ab'), argument_by_name(loss, 'ae'), z.outputs[0], z.outputs[1])
    
    # currently the err evaluation is too slow, so do it with low frequency
    class MyProgressPrinter(C.ProgressPrinter):
        def __init__(self, *kargs):
            super(MyProgressPrinter, self).__init__(*kargs)
            self.count = 0

        def on_write_training_summary(self, *kargs):
            super(MyProgressPrinter, self).on_write_training_summary(*kargs)
            self.count += 1
            if self.count % 20 == 0:
                print("Test error:", err.forward(mb_source.next_minibatch(1, input_map)))

    progress_writers = [MyProgressPrinter()]

    minibatch_size = 2048
    lr_schedule = C.learning_rate_schedule(0.000001, C.UnitType.sample)
    momentum_time_constant = -minibatch_size/np.log(0.9)
    mm_schedule = C.momentum_as_time_constant_schedule(momentum_time_constant)
    optimizer = C.adam_sgd(z.parameters, lr_schedule, mm_schedule, unit_gain=False, low_memory=False) # should use adadelta
    
    trainer = C.Trainer(z, (loss, None), optimizer, progress_writers)

    C.training_session(
        trainer=trainer,
        mb_source = mb_source,
        mb_size = minibatch_size,
        var_to_stream = input_map,
        max_samples = 35600,
        progress_frequency=100
    ).train()

if __name__=='__main__':
    training()