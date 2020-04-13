import json
import numpy as np
from squad_utils import normalize_answer
from preprocess_utils import preprocess, tokenize, make_str, trim_empty


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

def convert_file(json_file, train_file, val_file=None, train_ratio = 1.0, is_test=False, sort_by_len=False):
    data=json.load(open(json_file,"r", encoding="utf-8"))['data']
    qcount=0
    acount=0
    no_answer=[]
    no_matches=[]
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
                        assert(paragraph['context'][start:start+len(answer)]==answer)
                        atokens = tokenize(answer, context_mode=True)
                        atokens = trim_empty(atokens)
                        if start not in c2w:
                            #print("ID: {0} Cannot find answer {1}, context modified?: {2}"
                            #      .format(id, answer, paragraph['context']==context))
                            no_answer.append(id)
                            continue
                        offset=c2w[start]
                        assert(arec['text']==paragraph['context'][start:start+len(arec['text'])])
                        if not any([(i+offset<len(ctokens) and atoken==ctokens[i+offset]) for i,atoken in enumerate(atokens)]):
                            #print("ID: {0} Tokens do not match: answer [{1}] answer tokens [{2}] context tokens [{3}]"
                            #            .format(id, answer, atokens, ctokens[offset:offset+len(atokens)]))
                            no_matches.append(id)
                            continue
                        assert(ans_str is None)
                        ans_str='\t'+str(offset)+'\t'+str(offset+len(atokens))+'\t'+' '.join(atokens)
                else:
                    ans_str=''
                if ans_str is not None:
                    slist.append(id + '\t' + title + '\t' + make_str(ctokens) + '\t' + ' '.join(qtokens) + '\t' +
                        ';'.join([normalize_answer(arec['text']) for arec in qrec['answers']]) + "\t" + context.replace("\n", " ") +
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

    print(qcount, "questions", acount, "answers", len(no_answer), "no answer", len(no_matches), "no matches")
    print("Max context size", max_context_size, "Max question size", max_question_size)
    #print(no_answer)
    #print(no_matches)


if __name__ == '__main__':
    #dir="d:/work/data/squad/"
    dir="./"
    convert_file(dir+"train-v1.1.json", dir+"full_train.tsv")
    convert_file(dir+"train-v1.1.json", dir+"full_train_sort.tsv", sort_by_len=True)
    convert_file(dir+"train-v1.1.json", dir+"train.tsv", dir+"val.tsv", 0.98)
    convert_file(dir+"train-v1.1.json", dir+"train_sort.tsv", dir+"val_sort.tsv", 0.98, sort_by_len=True)
    convert_file(dir+"dev-v1.1.json", dir+"dev.tsv", is_test=True)
    convert_file(dir+"dev-v1.1.json", dir+"dev_sort.tsv", is_test=True, sort_by_len=True)
