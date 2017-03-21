import cntk as C
import numpy as np
from helpers import *
import pickle
import importlib
import os

class Bidaf:
    def __init__(self, config_file):
        data_config = importlib.import_module(config_file).data_config
        model_config = importlib.import_module(config_file).model_config

        self.word_count_threshold = data_config['word_count_threshold']
        self.char_count_threshold = data_config['char_count_threshold']
        self.word_size = data_config['word_size']
        self.max_query_len = data_config['max_query_len']
        self.max_context_len = data_config['max_context_len']
        self.abs_path = os.path.dirname(os.path.abspath(__file__))
        pickle_file = os.path.join(self.abs_path, data_config['pickle_file'])

        with open(pickle_file, 'rb') as vf:
            known, self.vocab, self.chars = pickle.load(vf)

        self.wg_dim = known
        self.wn_dim = len(self.vocab) - known
        self.c_dim = len(self.chars) * self.word_size
        self.a_dim = 1

        self.hidden_dim = model_config['hidden_dim']
        self.convs = model_config['char_convs']
        self.dropout = model_config['dropout']
        self.char_emb_dim = model_config['char_emb_dim']
        self.highway_layers = model_config['highway_layers']

    def charcnn(self, x):
        conv_out = C.models.Sequential([
            C.layers.Embedding(self.char_emb_dim),
            C.layers.Convolution1D((5,), self.convs, activation=C.relu, init=C.glorot_uniform(), pad=[True], strides=1, bias=True, init_bias=True)])(x)
        return C.reduce_max(conv_out, axis=1) # workaround cudnn failure in GlobalMaxPooling

    def embed(self):
        # load glove
        npglove = np.zeros((self.wg_dim, self.hidden_dim), dtype=np.float32)
        with open(os.path.join(self.abs_path, 'glove.6B.100d.txt'), encoding='utf-8') as f:
            for line in f:
                parts = line.split()
                word = parts[0].lower()
                if word in self.vocab:
                    npglove[self.vocab[word],:] = np.asarray([float(p) for p in parts[1:]])
        glove = C.constant(npglove)
        nonglove = C.parameter(shape=(len(self.vocab) - self.wg_dim, self.hidden_dim), init=C.glorot_uniform())
        
        def func(wg, wn):
            return C.times(wg, glove) + C.times(wn, nonglove)
        return func

    def model(self):
        c = C.Axis.new_unique_dynamic_axis('c')
        q = C.Axis.new_unique_dynamic_axis('q')
        b = C.Axis.default_batch_axis()
        cgw = C.input_variable(self.wg_dim, dynamic_axes=[b,c], is_sparse=True, name='cgw')
        cnw = C.input_variable(self.wn_dim, dynamic_axes=[b,c], is_sparse=True, name='cnw')
        qgw = C.input_variable(self.wg_dim, dynamic_axes=[b,q], is_sparse=True, name='qgw')
        qnw = C.input_variable(self.wn_dim, dynamic_axes=[b,q], is_sparse=True, name='qnw')
        cc = C.input_variable(self.c_dim, dynamic_axes=[b,c], name='cc')
        qc = C.input_variable(self.c_dim, dynamic_axes=[b,q], name='qc')
        ab = C.input_variable(self.a_dim, dynamic_axes=[b,c], name='ab')
        ae = C.input_variable(self.a_dim, dynamic_axes=[b,c], name='ae')

        input_chars = C.placeholder_variable(shape=(self.c_dim,))
        input_glove_words = C.placeholder_variable(shape=(self.wg_dim,))
        input_nonglove_words = C.placeholder_variable(shape=(self.wn_dim,))

        # we need to reshape because GlobalMaxPooling/reduce_max is retaining a trailing singleton dimension
        # todo GlobalPooling/reduce_max should have a keepdims default to False
        embedded = C.splice(
            self.embed()(input_glove_words, input_nonglove_words), 
            C.reshape(self.charcnn(C.reshape(input_chars, (C.InferredDimension, self.word_size))), self.convs))

        input_layers = C.layers.Sequential([
            HighwayNetwork(dim=(embedded.shape[0]), highway_layers=self.highway_layers),
            C.Dropout(self.dropout),
            OptimizedRnnStack(self.hidden_dim, bidirectional=True),
        ])
        
        q_emb = embedded.clone(C.CloneMethod.share, dict(zip(embedded.placeholders, [qgw,qnw,qc])))
        c_emb = embedded.clone(C.CloneMethod.share, dict(zip(embedded.placeholders, [cgw,cnw,cc])))

        q_processed = input_layers(q_emb) # synth_embedding for query
        c_processed = input_layers(c_emb) # synth_embedding for context

        #convert query's sequence axis to static
        qvw, qvw_mask = ValueWindow(self.max_query_len, C.Axis.new_leading_axis())(q_processed)

        #qvw_c = C.reduce_sum(qvw, 1)
        #qvw_c = print_node(qvw_c)
        #qvw = C.times(qvw_c, np.ones((1,200))) + qvw

        # This part deserves some explanation
        # It is the attention layer
        # In the paper they use a 6 * dim dimensional vector
        # here we split it in three parts because the different parts
        # participate in very different operations
        # so W * [h; u; h.* u] becomes w1 * h + w2 * u + w3 * (h.*u)
        ws1 = C.parameter(shape=(2 * self.hidden_dim, 1), init=C.glorot_uniform())
        ws2 = C.parameter(shape=(2 * self.hidden_dim, 1), init=C.glorot_uniform())
        ws3 = C.parameter(shape=(1, 2 * self.hidden_dim), init=C.glorot_uniform())
        att_bias = C.parameter(shape=(), init=0)

        wh = C.times (c_processed, ws1)
        wu = C.reshape(C.times (qvw, ws2), (self.max_query_len,))
        whu = C.times_transpose(c_processed, C.sequence.broadcast_as(C.element_times (qvw, ws3), c_processed))
        S = wh + whu + C.sequence.broadcast_as(wu, c_processed) + att_bias
        # mask out values outside of Query, and fill in gaps with -1e+30 as neutral value for both reduce_log_sum_exp and reduce_max
        qvw_mask_expanded = C.sequence.broadcast_as(C.reshape(qvw_mask, (self.max_query_len,)), c_processed)
        S = C.element_select(qvw_mask_expanded, S, C.constant(-1e+30, S.shape))
        q_logZ = C.reshape(C.reduce_log_sum_exp(S),(1,))
        q_attn = C.reshape(C.exp(S - q_logZ),(-1,1))
        #q_attn = print_node(q_attn)
        utilde = C.reshape(C.reduce_sum(C.sequence.broadcast_as(qvw, q_attn) * q_attn, axis=0),(-1))

        max_col = C.reduce_max(S)
        c_logZ = C.layers.Fold(C.log_add_exp, initial_state=C.constant(-1e+30, max_col.shape))(max_col)
        c_attn = C.exp(max_col - C.sequence.broadcast_as(c_logZ, max_col))
        htilde = C.sequence.reduce_sum(c_processed * c_attn)

        Htilde = C.sequence.broadcast_as(htilde, c_processed)
        att_context = C.splice(c_processed, utilde, c_processed * utilde,  c_processed * Htilde)

        #modeling layer
        mod_context = OptimizedRnnStack(self.hidden_dim, num_layers=2, bidirectional=True)(att_context)

        #output layer
        start_logits = C.layers.Dense(1)(C.splice(mod_context, att_context))
        start_hardmax = seq_hardmax(start_logits)
        start_prob = C.softmax(start_logits)
        att_mod_ctx = C.sequence.reduce_sum(mod_context * start_prob)
        att_mod_ctx_expanded = C.sequence.broadcast_as(att_mod_ctx, att_context)
        end_input = C.splice(att_context, mod_context, att_mod_ctx_expanded, mod_context * att_mod_ctx_expanded)
        m2 = OptimizedRnnStack(self.hidden_dim, bidirectional=True)(end_input)
        end_logits = C.layers.Dense(1)(C.splice(att_context, m2))

        start_loss = seq_loss(start_logits, ab)
        end_loss = seq_loss(end_logits, ae)
        end_hardmax = seq_hardmax(end_logits)

        return C.combine([start_hardmax, end_hardmax]), start_loss + end_loss

    def f1_score(self, ab, ae, oab, oae):
        answers = C.splice(ab, C.sequence.delay(ae), oab, C.sequence.delay(oae)) # make end non-inclusive
        answers_prop = C.Recurrence(C.element_max, go_backwards=False)(answers)
        ans_gt = C.element_min(C.slice(answers_prop, 0, 0, 1), (1 - C.slice(answers_prop, 0, 1, 2)))
        ans_out = C.element_min(C.slice(answers_prop, 0, 2, 3), (1 - C.slice(answers_prop, 0, 3, 4)))
        common = ans_gt * ans_out
        metric = C.layers.Fold(C.plus)(C.splice(ans_gt, ans_out, common))
        gt_len = C.slice(metric, 0, 0, 1)
        test_len = C.slice(metric, 0, 1, 2)
        common_len = C.slice(metric, 0, 2, 3)
        precision = common_len / test_len
        recall = common_len / gt_len
        return precision * recall * 2 / (precision + recall)