#!/usr/bin/env python

# This script takes a list of dictionary files and a plain text utf-8 file and converts this text input file to CNTK text format.
#
# The input text file must contain N streams per line (N TAB-separated "columns") and should be accompanied by N dictionary files.
# The input text file must be in the following form:
#    text1 TAB text2 TAB ... TAB textN
#    .....
# where each line represents one sequence across all N input streams.
# Each text consists of one or more space-separated word tokens (samples).
#
# Dictionary files are text files that are required to be specified for all streams,
# so the #dictionaries = #columns in the input file.
# A dictionary contains a single token per line. The zero-based line number becomes the numeric index
# of the token in the output CNTK text format file.

# Example usage (i.e. for PennTreebank files):
# 1)
#    sed -e 's/^<\/s> //' -e 's/ <\/s>$//' < en.txt > en.txt1
#    sed -e 's/^<\/s> //' -e 's/ <\/s>$//' < fr.txt > fr.txt1
#    paste en.txt1 fr.txt1 | txt2ctf.py --map en.dict fr.dict > en-fr.ctf
#
# 2) (assuming that the current dir is [cntk root]/Examples/SequenceToSequence/CMUDict/Data/)
# sed -e 's/<s\/>/<\/s>\t<s>/' < cmudict-0.7b.train-dev-1-21.txt `#this will replace every '<s/>' with '</s>[tab]<s>'` |\
# python ../../../../Scripts/txt2ctf.py --map cmudict-0.7b.mapping cmudict-0.7b.mapping > cmudict-0.7b.train-dev-1-21.ctf
#

import sys
import argparse
import re

def get_vocab(inputs):
    vocab={}
    for input in inputs:
      for line in input:
        l=line.rstrip().split()
        for w in l:
          try:
            vocab[w] = vocab[w] + 1
          except KeyError:
            vocab[w] = 1
    sorted_vocab = [i[0] for i in sorted(vocab.items(), key=lambda x:-x[1])]
    sorted_vocab = [(w,i) for i,w in enumerate(sorted_vocab)]
    g=open('vocab.txt','w',encoding='utf-8')
    for w,i in sorted_vocab:
      print("{}\t{}".format(w,i),file=g)
    g.close()
    return dict(sorted_vocab)

def convert(input, output, dict):
    for index, line in enumerate(input):
        line = line.rstrip('\r\n')
        words = line.split()
        for w1,w2 in zip(words[:-1],words[1:]):
            print("{}\t|S0 {}:1 |# {}\t|S1 {}:1 |# {}".format(index,dict[w1],w1,dict[w2],w2), file=output)
    input.close()
    output.close()
                
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transforms text file into CNTK text format.")
    parser.add_argument('--input', help='Name of the inputs files, stdin if not given', default="", nargs="*", required=False)
    args = parser.parse_args()

    # creating inputs
    inputs = [sys.stdin]
    if len(args.input) != 0:
        inputs = [open(i, encoding="utf-8") for i in args.input]
        
    dictionary = get_vocab(inputs)
    
    if len(args.input) != 0:
        inputs = [open(i, encoding="utf-8") for i in args.input]
        outputs = [open(i+'.ctf', 'w', encoding="utf-8") for i in args.input]

    for i in range(len(args.input)):    
      convert(inputs[i], outputs[i], dictionary)

