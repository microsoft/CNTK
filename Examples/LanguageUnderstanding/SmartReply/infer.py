import cntk as C
import pickle
import scipy.sparse
import numpy as np
from common import encoder_type, options

def load_trained_model(path, encoder_type='LSTM'):
    z = C.Function.load(path)
    msg_stack = C.as_composite(z.root_function.inputs[0].owner)
    reply_stack = C.as_composite(z.root_function.inputs[1].owner)
    return msg_stack, reply_stack


def featurize(sentence, vocab, threshold):
    sentence = sentence.strip().replace("|", "vbar")
    words = sentence.split()
    unk = vocab['UNK']
    ids = [vocab['BOS']] + [vocab[word] if wfreq[word] > threshold else unk for word in words] + [vocab['EOS']]
    return ids


def precompute_response_embeddings(reply_stack, responses, vocab, threshold):
    batch = []
    V = len(vocab)
    reply_var = reply_stack.arguments[0]
    for r in responses:
        col_ind = featurize(r, vocab, threshold)
        n = len(col_ind)
        data = np.ones(n,dtype=np.float32)
        row_ind = np.arange(n, dtype=np.int32)
        batch.append(scipy.sparse.csr_matrix((data, (row_ind, col_ind)), shape=(n, V+1)))
    response_embeddings = reply_stack.eval({reply_var:batch}, as_numpy=False)
    return C.constant(response_embeddings)


def score(score_net, messages, vocab, threshold):
    batch = []
    V = len(vocab)
    var = score_net.arguments[0]
    for r in messages:
        col_ind = featurize(r, vocab, threshold)
        n = len(col_ind)
        data = np.ones(n,dtype=np.float32)
        row_ind = np.arange(n, dtype=np.int32)
        batch.append(scipy.sparse.csr_matrix((data, (row_ind, col_ind)), shape=(n, V+1)))
    return score_net.eval({var:batch})



with open('words.pkl', 'rb') as pkl:
    vocab, wfreq, loaded_threshold = pickle.load(pkl)

for i in range(1,17):
    path = 'model.%s.%.2d.cnt'%(encoder_type,i)
    msg_stack, reply_stack = load_trained_model(path)
    msgs = ['bad news', 'good news', 'wanna grub some lunch ?']
    responses = ['thank you', 'this is great', 'meet me at the theater', 'oh no', 'oh yeah', 'i d love that']
    print('##### %.2d #####'%i)
    response_embeddings = precompute_response_embeddings(reply_stack, responses, vocab, loaded_threshold)
    score_net = C.times_transpose(msg_stack, response_embeddings)
    replies = score(score_net, msgs, vocab, loaded_threshold)
    for msg, rpl in zip(msgs, replies):
        print(msg)
        for x in rpl.argsort()[::-1]:
            print(responses[x], rpl[x])
        print('-----')
