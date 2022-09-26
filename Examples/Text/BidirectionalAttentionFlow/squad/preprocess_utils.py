import nltk
import re

def preprocess(s):
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


# map from token to char offset
def w2c_map(s, words):
    w2c = []
    rem = s
    offset = 0
    for i, w in enumerate(words):
        if w=='<NULL>':
            w=''
        if w=='"':
            # this can be either ""', "''", "``"
            maxlen=len(s)

            def getidx(seq):
                idx = rem.find(seq)
                return idx if idx >= 0 else maxlen

            cidx = getidx('"')
            mindouble=min(getidx("''"),getidx("``"))
            if mindouble<cidx:
                cidx=mindouble
                w="''" # does not matter which one, since they are the same length
            else:
                cidx=cidx if cidx<maxlen else -1
        else:
            cidx = rem.find(w)
        assert (cidx >= 0)
        w2c.append(cidx + offset)
        offset += cidx + len(w)
        rem = rem[cidx + len(w):]
    return w2c