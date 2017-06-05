import nltk
import json
import numpy as np
import re
from squad_utils import normalize_answer

def preprocess(s, add_quote=True):
    rep='"'+(" " if add_quote else "")
    return s.replace("''", '" ').replace("``", '" ')

def tokenize(s, context_mode=False ):
    nltk_tokens=[t.replace("''", '"').replace("``", '"') for t in nltk.word_tokenize(s)]
    additional_separators = (
             "-", "\u2212", "\u2014", "\u2013", "/", "~", '"', "'", "\u201C", "\u2019", "\u201D", "\u2018", "\u00B0")
    tokens = []
    for token in nltk_tokens:
        # TODO: extra sepatators are only processed for context. It is done to repeat conditions in BiDAF to
        # use its variable values. However, it does not necessary make sense to do that for trained model
        # Also empty tokens are preserved for BiDAF compat, should probably be thrown away
        tokens.extend([t for t in (re.split("([{}])".format("".join(additional_separators)), token)
                                   if context_mode else [token])])
    assert(not any([t=='<NULL>' for t in tokens]))
    assert(not any([' ' in t for t in tokens]))
    assert (not any(['\t' in t for t in tokens]))
    return tokens

# map from character offset to token offset
def c2w_map(s, words):
    c2w={}
    rem=s
    offset=0
    for i,w in enumerate(words):
        if len(w)>0:
            cidx=rem.find(w)
            assert(cidx>=0)
            c2w[cidx+offset]=i
            offset+=cidx + len(w)
            rem=rem[cidx + len(w):]
    return c2w

def make_str(tokens):
    return ' '.join(['<NULL>' if t=='' else t for t in tokens])

def trim_empty(tokens):
    trimmed=[]
    good_len=0
    for t in tokens:
        if t =='' and good_len==0:
            continue
        trimmed.append(t)
        if t != '':
            good_len=len(trimmed)
    return trimmed[:good_len]

def convert_file(json_file, train_file, val_file=None, train_ratio = 1.0, is_test=False, sort_by_len=False):
    data=json.load(open(json_file,"r", encoding="utf-8"))['data']
    qcount=0
    acount=0
    no_answer=0
    no_matches=0
    slist=[]
    max_context_size=0
    max_question_size=0
    for article in data:
        title=article['title']
        for paragraph in article['paragraphs']:
            context=preprocess(paragraph['context'])
            ctokens=tokenize(context, context_mode=True)
            max_context_size=max(max_context_size, len(ctokens))
            c2w = c2w_map(context, ctokens)
            assert('\t' not in context)
            for qrec in paragraph['qas']:
                id=qrec['id']
                question=preprocess(qrec['question'])
                assert('\t' not in question)
                qtokens=tokenize(question)
                max_question_size=max(len(qtokens), max_question_size)
                ans_str = None
                qcount+=1
                acount+=len(qrec['answers'])
                if not is_test:
                    # training format - only one answer is expected, and we need to identify offsets
                    for arec in qrec['answers']:
                        start=int(arec['answer_start'])
                        answer = preprocess(arec['text'])
                        atokens = tokenize(answer, context_mode=True)
                        atokens = trim_empty(atokens)
                        if start not in c2w:
                            #print("Cannot find answer", answer, paragraph['context']==context)
                            no_answer+=1
                            continue
                        offset=c2w[start]
                        assert(arec['text']==paragraph['context'][start:start+len(arec['text'])])
                        if not any([(i+offset<len(ctokens) and atoken==ctokens[i+offset]) for i,atoken in enumerate(atokens)]):
                            #print("Tokens do not match", '<<'+answer+'>>', atokens, ctokens[offset:offset+len(atokens)])
                            no_matches+=1
                            continue
                        assert(ans_str is None)
                        ans_str=str(offset)+'\t'+str(offset+len(atokens))+'\t'+' '.join(atokens)
                else:
                    # testing format - can have multiple answers, offsets are not used
                    ans_str=';'.join([normalize_answer(arec['text']) for arec in qrec['answers']]) + "\t" + context.replace("\n"," ")
                if ans_str is not None:
                    slist.append(id + '\t' + title + '\t' + make_str(ctokens) + '\t' + ' '.join(qtokens) + '\t' +
                        ans_str + '\n')

    r=np.random.RandomState(2017)
    r.shuffle(slist)

    def get_len(s):
        parts = s.split('\t')
        return (len(parts[2]), len(parts[3]))

    if val_file is None:
        train_data=slist
    else:
        train_data=[]
        val_data=[]
        for s in slist:
            (train_data if r.uniform()<train_ratio else val_data).append(s)
        if sort_by_len:
            val_data.sort(key=get_len)
        with open(val_file, "w", encoding="utf-8") as val:
            print("Writing", val_file)
            val.writelines(val_data)

    if sort_by_len:
        train_data.sort(key=get_len)
    print("Writing", train_file)
    with open(train_file, "w", encoding="utf-8") as train:
        train.writelines(train_data)

    print(qcount, "questions", acount, "answers", no_answer, "no answer", no_matches, "no matches")
    print("Max context size", max_context_size, "Max question size", max_question_size)






#dir="d:/work/data/squad/"
dir="./"

convert_file(dir+"train-v1.1.json", dir+"full_train.tsv")
convert_file(dir+"train-v1.1.json", dir+"full_train_sort.tsv", sort_by_len=True)
convert_file(dir+"train-v1.1.json", dir+"train.tsv", dir+"val.tsv", 0.98)
convert_file(dir+"train-v1.1.json", dir+"train_sort.tsv", dir+"val_sort.tsv", 0.98, sort_by_len=True)
convert_file(dir+"dev-v1.1.json", dir+"dev.tsv", is_test=True)
convert_file(dir+"dev-v1.1.json", dir+"dev_sort.tsv", is_test=True, sort_by_len=True)



