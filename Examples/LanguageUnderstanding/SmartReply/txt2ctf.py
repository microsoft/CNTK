from collections import defaultdict, Counter
from itertools import count, chain
import pickle
import random
import argparse


def ngrams_in_stream(s, k=3):
    m = len(s)
    for n in range(1, k + 1):
        for i in range(m+1-n):
            yield ' '.join(s[i:i + n])

def ngrams(s, k=3):
    streams = s.split('\t')
    for stream in streams:
        words = stream.split()
        for ngram in ngrams_in_stream(words, k):
            yield ngram


def save_vocab(file, generator, threshold):
    vocab = defaultdict(count().__next__)
    bos = vocab['BOS']
    eos = vocab['EOS']
    unk = vocab['UNK']
    with open(file, encoding='utf-8') as inp:
        wfreq = Counter(chain.from_iterable(map(generator, inp)))
    dummy = [vocab[word] for word, freq in wfreq.most_common() if freq > threshold]
    print(len(vocab))
    vocab.default_factory = None
    return vocab, wfreq, threshold


def get_vocab(pkl_path, input_path, threshold, generator):
    try:
        with open('words.pkl', 'rb') as pkl:
            vocab, wfreq, loaded_threshold = pickle.load(pkl)
        if loaded_threshold != threshold:
            raise Exception('non matching threshold')
    except:
        print('reparsing')
        (vocab, wfreq, threshold) = save_vocab(input_path, generator, threshold)
        with open('words.pkl', 'wb') as pkl:
            pickle.dump((vocab, wfreq, threshold), pkl)
    return vocab, wfreq, threshold

if __name__ == '__main__':

    random.seed(98052)
    threshold = 50
    train_fraction = 0.8
    valid_fraction = 0.9

    parser = argparse.ArgumentParser(description='Preprocess txt to ctf')
    parser.add_argument('input', type=str, help='input path')
    parser.add_argument('--lstm', dest='type', action='store_const',
                        const='lstm', default='ff',
                        help='process to lstm or ff format (default: ff)')

    args = parser.parse_args()
    print(args.type)
    if args.type == 'lstm':
        vocab, wfreq, threshold = get_vocab('words.pkl', args.input, threshold, str.split)
        out_train = 'train.seq.ctf'
        out_valid = 'valid.seq.ctf'
        out_test = 'test.seq.ctf'
    elif args.type == 'ff':
        vocab, wfreq, threshold = get_vocab('words.ff.pkl', args.input, threshold, ngrams)
        out_train = 'train.ff.ctf'
        out_valid = 'valid.ff.ctf'
        out_test = 'test.ff.ctf'
    else:
        raise ValueError('Unknown type to process')

    bos = vocab['BOS']
    eos = vocab['EOS']
    unk = vocab['UNK']

    with open(args.input, encoding='utf-8') as inp:
        with open(out_train, 'w', encoding='utf-8') as trn:
            with open(out_valid, 'w', encoding='utf-8') as val:
                with open(out_test, 'w', encoding='utf-8') as tst:
                    for lineno,line in enumerate(inp):
                        rand = random.random()
                        if rand < train_fraction:
                            out = trn
                        elif rand < valid_fraction:
                            out = val
                        else:
                            out = tst
                        context, answer = line.strip().replace("|","vbar").split('\t')
                        if args.type == 'lstm':
                            cwordlist = context.split()
                            awordlist = answer.split()
                            cids = [bos] + [vocab[word] if wfreq[word] > threshold else unk for word in cwordlist] + [eos]
                            aids = [bos] + [vocab[word] if wfreq[word] > threshold else unk for word in awordlist] + [eos]
                            cwordlist = ['BOS'] + cwordlist + ['EOS']
                            awordlist = ['BOS'] + awordlist + ['EOS']
                            difference = len(cids) - len(aids)
                            for cid, cw, aid, aw in zip(cids, cwordlist, aids, awordlist):
                                out.write('%d |c %d:1 |# %s |a %d:1 |# %s\n'%(lineno, cid, cw, aid, aw))
                                #out.write('%d |c %d:1 |a %d:1\n' % (lineno, cid, aid))
                            if difference > 0:
                                for cid, cw in zip(cids[-difference:], cwordlist[-difference:]):
                                    out.write('%d |c %d:1 |# %s\n' % (lineno, cid, cw))
                                    #out.write('%d |c %d:1\n' % (lineno, cid))
                            elif difference < 0:
                                for aid, aw in zip(aids[difference:], awordlist[difference:]):
                                    out.write('%d |a %d:1 |# %s\n' % (lineno, aid, aw))
                                    #out.write('%d |a %d:1\n' % (lineno, aid))
                        else:
                            cwordlist = list(ngrams_in_stream(context.split()))
                            awordlist = list(ngrams_in_stream(answer.split()))
                            cids = Counter(vocab[word] if wfreq[word] > threshold else unk for word in cwordlist)
                            aids = Counter(vocab[word] if wfreq[word] > threshold else unk for word in awordlist)
                            sumcids = float(sum(cids.values()))
                            sumaids = float(sum(aids.values()))
                            bufc = ['|c']+['%d:%.4f' % (cid, cids[cid] / sumcids) for cid in cids]
                            bufa = ['|a']+['%d:%.4f' % (aid, aids[aid] / sumaids) for aid in aids]
                            out.write('%s %s\n'%(' '.join(bufc),' '.join(bufa)))