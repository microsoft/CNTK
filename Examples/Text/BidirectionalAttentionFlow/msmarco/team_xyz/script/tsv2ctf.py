from collections import defaultdict
from itertools import count, zip_longest
from config import *
import pickle
import numpy as np

word_count_threshold = data_config['word_count_threshold']
char_count_threshold = data_config['char_count_threshold']
word_size = data_config['word_size']

sanitize = str.maketrans({"|": None, "\n": None})
tsvs = 'train', 'dev', 'test'
unk = '<UNK>'
pad = ''
EMPTY_TOKEN = '<NULL>'
# pad (or trim) to word_size characters
pad_spec = '{0:<%d.%d}' % (word_size, word_size)

def populate_dicts(files):
    vocab = defaultdict(count().__next__)
    chars = defaultdict(count().__next__)
    wdcnt = defaultdict(int)
    chcnt = defaultdict(int)
    test_wdcnt = defaultdict(int) # all glove words in test/dev should be added to known, but non-glove words in test/dev should be kept unknown

    # count the words and characters to find the ones with cardinality above the thresholds
    for f in files:
        with open('%s.tsv' % f, 'r', encoding='utf-8') as input_file:
            for line in input_file:
                if 'test' in f:
                    uid, title, context, query = line.split('\t')
                else:
                    uid, title, context, query, answer, raw_context, begin_answer, end_answer, raw_answer = line.split('\t')
                tokens = context.split(' ')+query.split(' ')
                if 'train' in f:
                    for t in tokens:
                        wdcnt[t.lower()] += 1
                        for c in t: chcnt[c] += 1
                else:
                    for t in tokens:
                        test_wdcnt[t.lower()] += 1

    # add all words that are both in glove and the vocabulary first
    with open('glove.6B.100d.txt', encoding='utf-8') as f:
        for line in f:
            word = line.split()[0].lower()
            if wdcnt[word] >= 1 or test_wdcnt[word] >= 1: # polymath adds word to dict regardless of word_count_threshold when it's in GloVe
                _ = vocab[word]
    known =len(vocab)

    # add the special markers
    _ = vocab[pad]
    _ = vocab[unk]
    _ = chars[pad]
    _ = chars[unk]

    #finally add all words that are not in yet
    _  = [vocab[word] for word in wdcnt if word not in vocab and wdcnt[word] > word_count_threshold]
    _  = [chars[c]    for c    in chcnt if c    not in chars and chcnt[c]    > char_count_threshold]

    # return as defaultdict(int) so that new keys will return 0 which is the value for <unknown>
    return known, defaultdict(int, vocab), defaultdict(int, chars)

def tsv_iter(line, vocab, chars, is_test=False, misc={}):
    unk_w = vocab[unk]
    unk_c = chars[unk]

    if is_test:
        uid, title, context, query = line.split('\t')
        answer = ''
        begin_answer, end_answer = '0', '1'
    else:
        uid, title, context, query, answer, raw_context, begin_answer, end_answer, raw_answer = line.split('\t')
        #uid, title, context, query, begin_answer, end_answer, answer = line.split('\t')

    ctokens = context.split(' ')
    qtokens = query.split(' ')

    #replace EMPTY_TOKEN with ''
    ctokens = [t if t != EMPTY_TOKEN else '' for t in ctokens]
    qtokens = [t if t != EMPTY_TOKEN else '' for t in qtokens]


    cwids = [vocab.get(t.lower(), unk_w) for t in ctokens]
    qwids = [vocab.get(t.lower(), unk_w) for t in qtokens]
    ccids = [[chars.get(c, unk_c) for c in t][:word_size] for t in ctokens] #clamp at word_size
    qcids = [[chars.get(c, unk_c) for c in t][:word_size] for t in qtokens]

    ba, ea = int(begin_answer), int(end_answer) - 1 # the end from tsv is exclusive
    if ba > ea:
        raise ValueError('answer problem with input line:\n%s' % line)

    baidx = [0 if i != ba else 1 for i,t in enumerate(ctokens)]
    eaidx = [0 if i != ea else 1 for i,t in enumerate(ctokens)]
    
    atokens = answer.split(' ')
    
    if not is_test and sum(eaidx) == 0:
        raise ValueError('problem with input line:\n%s' % line)
    
    if is_test and misc.keys():
        misc['answer'] += [answer]
        misc['rawctx'] += [context]
        misc['ctoken'] += [ctokens]

    return ctokens, qtokens, atokens, cwids, qwids, baidx, eaidx, ccids, qcids

def tsv_to_ctf(f, g, vocab, chars, is_test):
    print("Known words: %d" % known)
    print("Vocab size: %d" % len(vocab))
    print("Char size: %d" % len(chars))
    for lineno, line in enumerate(f):
        ctokens, qtokens, atokens, cwids, qwids,  baidx,   eaidx, ccids, qcids = tsv_iter(line, vocab, chars, is_test)

        for     ctoken,  qtoken,  atoken,  cwid,  qwid,   begin,   end,   ccid,  qcid in zip_longest(
                ctokens, qtokens, atokens, cwids, qwids,  baidx,   eaidx, ccids, qcids):
            out = [str(lineno)]
            if ctoken is not None:
                out.append('|# %s' % pad_spec.format(ctoken.translate(sanitize)))
            if qtoken is not None:
                out.append('|# %s' % pad_spec.format(qtoken.translate(sanitize)))
            if atoken is not None:
                out.append('|# %s' % pad_spec.format(atoken.translate(sanitize)))
            if begin is not None:
                out.append('|ab %3d' % begin)
            if end is not None:
                out.append('|ae %3d' % end)
            if cwid is not None:
                if cwid >= known:
                    out.append('|cgw {}:{}'.format(0, 0))
                    out.append('|cnw {}:{}'.format(cwid - known, 1))
                else:
                    out.append('|cgw {}:{}'.format(cwid, 1))
                    out.append('|cnw {}:{}'.format(0, 0))
            if qwid is not None:
                if qwid >= known:
                    out.append('|qgw {}:{}'.format(0, 0))
                    out.append('|qnw {}:{}'.format(qwid - known, 1))
                else:
                    out.append('|qgw {}:{}'.format(qwid, 1))
                    out.append('|qnw {}:{}'.format(0, 0))
            if ccid is not None:
                outc = ' '.join(['%d' % c for c in ccid+[0]*max(word_size - len(ccid), 0)])
                out.append('|cc %s' % outc)
            if qcid is not None:
                outq = ' '.join(['%d' % c for c in qcid+[0]*max(word_size - len(qcid), 0)])
                out.append('|qc %s' % outq)
            g.write('\t'.join(out))
            g.write('\n')

if __name__=='__main__':
    try:
        known, vocab, chars = pickle.load(open('vocabs.pkl', 'rb'))
    except:
        known, vocab, chars = populate_dicts(tsvs)
        f = open('vocabs.pkl', 'wb')
        pickle.dump((known, vocab, chars), f)
        f.close()

    for tsv in tsvs:
        with open('%s.tsv' % tsv, 'r', encoding='utf-8') as f:
            with open('%s.ctf' % tsv, 'w', encoding='utf-8') as g:
                tsv_to_ctf(f, g, vocab, chars, tsv == 'test')
