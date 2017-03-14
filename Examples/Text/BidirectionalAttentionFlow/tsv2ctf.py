from collections import defaultdict
from itertools import count, zip_longest
import pickle
import numpy as np

char_count_threshold = 50
word_count_threshold = 10
word_size = 20

sanitize = str.maketrans({"|": None, "\n": None})
tsvs = 'train', 'dev', 'val'
unk = '<unk>'
eos = '</s>'
pad = ' '  # used for padding has to be single char and match with format specifier below
# pad (or trim) to word_size characters
pad_spec = '{0:<%d.%d}' % (word_size, word_size)

def populate_dicts(files):
    vocab = defaultdict(count().__next__)
    chars = defaultdict(count().__next__)
    wdcnt = defaultdict(int)
    chcnt = defaultdict(int)

    # count the words and characters to find the ones with cardinality above the thresholds
    for f in files:
        with open('%s.tsv' % f, 'r', encoding='utf-8') as input_file:
            for line in input_file:
                uid, title, context, query, begin_answer, end_answer, answer = line.split('\t')
                tokens = context.split(' ')+query.split(' ')
                for t in tokens:
                    wdcnt[t.lower()] += 1
                    for c in t: chcnt[c] += 1

    # add the special markers first, so <unknown> is 0
    _ = vocab[unk]
    _ = vocab[eos]
    _ = chars[unk]
    _ = chars[pad]

    # add all words that are both in glove and the vocabulary first
    with open('glove.6B.100d.txt', encoding='utf-8') as f:
        for line in f:
            word = line.split()[0].lower()
            if wdcnt[word] >= 1: # polymath adds word to dict regardless of word_count_threshold when it's in GloVe
                _ = vocab[word]
    known =len(vocab)

    #finally add all words that are not in yet
    _  = [vocab[word] for word in wdcnt if word not in vocab and wdcnt[word] >= word_count_threshold]
    _  = [chars[c]    for c    in chcnt if c    not in chars and chcnt[c]    >= char_count_threshold]

    # return as defaultdict(int) so that new keys will return 0 which is the value for <unknown>
    return known, defaultdict(int, vocab), defaultdict(int, chars)

def tsv_to_ctf(f, g, vocab, chars, is_test):
    for lineno, line in enumerate(f):
        if is_test:
            uid, title, context, query, answer, other = line.split('\t')
            begin_answer, end_answer = '0', '1'
        else:
            uid, title, context, query, begin_answer, end_answer, answer = line.split('\t')
        ctokens = context.split(' ')
        ctokens.append(eos)
        qtokens = query.split(' ')
        atokens = answer.split(' ')
        cwids = [vocab[t.lower()] for t in ctokens]
        qwids = [vocab[t.lower()] for t in qtokens]
        ccids = [[chars[c] for c in pad_spec.format(t)] for t in ctokens]
        qcids = [[chars[c] for c in pad_spec.format(t)] for t in qtokens]
        ba, ea = int(begin_answer), int(end_answer) - 1 # the end from tsv is exclusive
        baidx = [0 if i != ba else 1 for i,t in enumerate(ctokens)]
        eaidx = [0 if i != ea else 1 for i,t in enumerate(ctokens)]
        if sum(eaidx) == 0:
            raise ValueError('problem with input line:\n%s' % line)
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
                out.append('|cw %d:1' % cwid)
            if qwid is not None:
                out.append('|qw %d:1' % qwid)
            if ccid is not None:
                outc = ' '.join(['%d:1' % (word_size * c + i) for i, c in enumerate(ccid)])
                out.append('|cc %s' % outc)
            if qcid is not None:
                outq = ' '.join(['%d:1' % (word_size * c + i) for i, c in enumerate(qcid)])
                out.append('|qc %s' % outq)
            g.write('\t'.join(out))
            g.write('\n')

try:
    known, vocab, chars = pickle.load(open('vocabs.pkl', 'rb'))
except:
    known, vocab, chars = populate_dicts((tsvs[0],))
    f = open('vocabs.pkl', 'wb')
    pickle.dump((known, vocab, chars), f)
    f.close()

for tsv in tsvs:
    with open('%s.tsv' % tsv, 'r', encoding='utf-8') as f:
        with open('%s.ctf' % tsv, 'w', encoding='utf-8') as g:
            tsv_to_ctf(f, g, vocab, chars, tsv == 'dev')
