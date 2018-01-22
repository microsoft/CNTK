import gzip
import json
import numpy as np
import nltk
import re
import bisect

def smith_waterman(tt,bb):
    # adapted from https://gist.github.com/radaniba/11019717

    # These scores are taken from Wikipedia.
    # en.wikipedia.org/wiki/Smith%E2%80%93Waterman_algorithm
    match    = 2
    mismatch = -1
    gap      = -1

    def calc_score(matrix, x, y, seq1, seq2):
        '''Calculate score for a given x, y position in the scoring matrix.
        The score is based on the up, left, and upper-left neighbors.
        '''
        similarity = match if seq1[x - 1] == seq2[y - 1] else mismatch

        diag_score = matrix[x - 1, y - 1] + similarity
        up_score   = matrix[x - 1, y] + gap
        left_score = matrix[x, y - 1] + gap

        return max(0, diag_score, up_score, left_score)


    def create_score_matrix(rows, cols, seq1, seq2):
        '''Create a matrix of scores representing trial alignments of the two sequences.
        Sequence alignment can be treated as a graph search problem. This function
        creates a graph (2D matrix) of scores, which are based on trial alignments. 
        The path with the highest cummulative score is the best alignment.
        '''
        score_matrix = np.zeros((rows,cols))

        # Fill the scoring matrix.
        max_score = 0
        max_pos   = None    # The row and columbn of the highest score in matrix.
        for i in range(1, rows):
            for j in range(1, cols):
                score = calc_score(score_matrix, i, j, seq1, seq2)
                if score > max_score:
                    max_score = score
                    max_pos   = (i, j)

                score_matrix[i, j] = score

        if max_pos is None:
            raise ValueError('cannot align %s and %s'%(' '.join(seq1)[:80],' '.join(seq2)))

        return score_matrix, max_pos

    def next_move(score_matrix, x, y):
        diag = score_matrix[x - 1, y - 1]
        up   = score_matrix[x - 1, y]
        left = score_matrix[x, y - 1]
        if diag >= up and diag >= left:     # Tie goes to the DIAG move.
            return 1 if diag != 0 else 0    # 1 signals a DIAG move. 0 signals the end.
        elif up > diag and up >= left:      # Tie goes to UP move.
            return 2 if up != 0 else 0      # UP move or end.
        elif left > diag and left > up:
            return 3 if left != 0 else 0    # LEFT move or end.
        else:
            # Execution should not reach here.
            raise ValueError('invalid move during traceback')

    def traceback(score_matrix, start_pos, seq1, seq2):
        '''Find the optimal path through the matrix.
        This function traces a path from the bottom-right to the top-left corner of
        the scoring matrix. Each move corresponds to a match, mismatch, or gap in one
        or both of the sequences being aligned. Moves are determined by the score of
        three adjacent squares: the upper square, the left square, and the diagonal
        upper-left square.
        WHAT EACH MOVE REPRESENTS
            diagonal: match/mismatch
            up:       gap in sequence 1
            left:     gap in sequence 2
        '''

        END, DIAG, UP, LEFT = range(4)
        x, y         = start_pos
        move         = next_move(score_matrix, x, y)
        while move != END:
            if move == DIAG:
                x -= 1
                y -= 1
            elif move == UP:
                x -= 1
            else:
                y -= 1
            move = next_move(score_matrix, x, y)

        return (x,y), start_pos

    rows = len(tt) + 1
    cols = len(bb) + 1

    # Initialize the scoring matrix.
    score_matrix, start_pos = create_score_matrix(rows, cols, tt, bb)
    
    # Traceback. Find the optimal path through the scoring matrix. This path
    # corresponds to the optimal local sequence alignment.
    (x,y), (w,z) = traceback(score_matrix, start_pos, tt, bb)
    return (x,w), (y,z), score_matrix[w][z]


def preprocess(s):
    return s.replace("''", '" ').replace("``", '" ')

def tokenize(s, context_mode=False ):
    nltk_tokens=[t.replace("''", '"').replace("``", '"') for t in nltk.word_tokenize(s)]
    additional_separators = (
             "-", "\u2212", "\u2014", "\u2013", "/", "~", '"', "'", "\u201C", "\u2019", "\u201D", "\u2018", "\u00B0")
    tokens = []
    for token in nltk_tokens:
        tokens.extend([t for t in (re.split("([{}])".format("".join(additional_separators)), token)
                                   if context_mode else [token])])
    assert(not any([t=='<NULL>' for t in tokens]))
    assert(not any([' ' in t for t in tokens]))
    assert (not any(['\t' in t for t in tokens]))
    return tokens

def trim_empty(tokens):
    return [t for t in tokens if t != '']

def convert(file, outfile, is_test):
    with gzip.open(file,'rb') as f:
        with open(outfile, 'w', encoding='utf-8') as out:
            for i,line in enumerate(f):
                j = json.loads(line.decode('utf-8'))
                p = j['passages']
            
                if j['query_type'] == 'description':
                    context = preprocess(' '.join([pp['passage_text'] for pp in p]))
                    ctokens = trim_empty(tokenize(context, context_mode=True))
                    normalized_context = ' '.join(ctokens)
                    nctokens = normalized_context.split()

                    query   = preprocess(j['query'])
                    qtokens =  trim_empty(tokenize(query))

                    if not is_test:
                        for a in j['answers']:
                            bad = False
                            answer = preprocess(a)
                            atokens = trim_empty(tokenize(answer, context_mode=True))
                            normalized_answer = ' '.join(atokens).lower()
                            normalized_context_lower = normalized_context.lower()
                            pos = normalized_context_lower.find(normalized_answer)
                            if pos >= 0:
                                # exact match; no need to run smith waterman
                                start = bisect.bisect(np.cumsum([1+len(t) for t in nctokens]), pos)
                                end = start + len(atokens)
                            else:
                                natokens = normalized_answer.split()
                                try:
                                    (start, end), (astart, aend), score = smith_waterman(normalized_context_lower.split(), natokens)
                                    ratio = 0.5 * score / min(len(nctokens), len(natokens))
                                    if ratio < 0.8:
                                        bad = True
                                except:
                                    bad = True
                            if not bad:
                                output = [str(j['query_id']), j['query_type'], ' '.join(nctokens),' '.join(qtokens),' '.join(nctokens[start:end]), normalized_context, str(start), str(end), normalized_answer]
                    else:
                        output = [str(j['query_id']), j['query_type'], ' '.join(nctokens),' '.join(qtokens)]
                    out.write("%s\n"%'\t'.join(output))

convert('train_v1.1.json.gz', 'train.tsv', False)
convert('dev_v1.1.json.gz', 'dev.tsv', False)
convert('test_public_v1.1.json.gz', 'test.tsv', True)
