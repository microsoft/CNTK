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

def convert(dictionaryStreams, inputs, output, unk, annotated):
    # create in memory dictionaries
    dictionaries = [{ line.rstrip('\r\n').strip():index for index, line in enumerate(dic) } for dic in dictionaryStreams]

    # convert inputs
    for input in inputs:
        sequenceId = 0
        for index, line in enumerate(input):
            line = line.rstrip('\r\n')
            columns = line.split("\t")
            if len(columns) != len(dictionaries):
                raise Exception("Number of dictionaries {0} does not correspond to the number of streams in line {1}:'{2}'"
                    .format(len(dictionaries), index, line))
            _convertSequence(dictionaries, columns, sequenceId, output, unk, annotated)
            sequenceId += 1

def _convertSequence(dictionaries, streams, sequenceId, output, unk, annotated):
    tokensPerStream = [[t for t in s.strip(' ').split(' ') if t != ""] for s in streams]
    maxLen = max(len(tokens) for tokens in tokensPerStream)

    # writing to the output file
    for sampleIndex in range(maxLen):
        output.write(str(sequenceId))
        for streamIndex in range(len(tokensPerStream)):
            if len(tokensPerStream[streamIndex]) <= sampleIndex:
                output.write("\t")
                continue
            token = tokensPerStream[streamIndex][sampleIndex]
            if unk is not None and token not in dictionaries[streamIndex]: # try unk symbol if specified
                token = unk
            if token not in dictionaries[streamIndex]:
                raise Exception("Token '{0}' cannot be found in the dictionary for stream {1}".format(token, streamIndex))
            value = dictionaries[streamIndex][token]
            output.write("\t|S" + str(streamIndex) + " "+ str(value) + ":1")
            if annotated:
                output.write(" |# " + re.sub(r'(\|(?!#))|(\|$)', r'|#', token))
        output.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transforms text file given dictionaries into CNTK text format.")
    parser.add_argument('--map', help='List of dictionaries, given in the same order as streams in the input files',
        nargs="+", required=True)
    parser.add_argument('--annotated', help='Whether to annotate indices with tokens. Default is false',
        choices=["True", "False"], default="False", required=False)
    parser.add_argument('--output', help='Name of the output file, stdout if not given', default="", required=False)
    parser.add_argument('--input', help='Name of the inputs files, stdin if not given', default="", nargs="*", required=False)
    parser.add_argument('--unk', help='Name fallback symbol for tokens not in dictionary (same for all columns)', default=None, required=False)
    args = parser.parse_args()

    # creating inputs
    inputs = [sys.stdin]
    if len(args.input) != 0:
        inputs = [open(i, encoding="utf-8") for i in args.input]

    # creating output
    output = sys.stdout
    if args.output != "":
        output = open(args.output, "w")

    convert([open(d, encoding="utf-8") for d in args.map], inputs, output, args.unk, args.annotated == "True")
    output.flush()
    if (output != sys.stdout):
        output.close()

#####################################################################################################
# Tests
#####################################################################################################
try:
    import StringIO
    stringio = StringIO.StringIO
except ImportError:
    from io import StringIO
    stringio = StringIO
try:
    import pytest
except ImportError:
    pass

def test_simpleSanityCheck():
    dictionary1 = stringio("hello\nmy\nworld\nof\nnothing\n")
    dictionary2 = stringio("let\nme\nbe\nclear\nabout\nit\n")
    input = stringio("hello my\tclear about\nworld of\tit let clear\n")
    output = stringio()

    convert([dictionary1, dictionary2], [input], output, None, False)

    expectedOutput = stringio()
    expectedOutput.write("0\t|S0 0:1\t|S1 3:1\n")
    expectedOutput.write("0\t|S0 1:1\t|S1 4:1\n")
    expectedOutput.write("1\t|S0 2:1\t|S1 5:1\n")
    expectedOutput.write("1\t|S0 3:1\t|S1 0:1\n")
    expectedOutput.write("1\t\t|S1 3:1\n")

    assert expectedOutput.getvalue() == output.getvalue()

def test_thatPipeSymbolIsEscaped():
    dictionary1 = stringio("|hello\nm|y\nworl|d\nof\nnothing|\n")
    dictionary2 = stringio("let|\nm|e\nb|#e\nclear\n||about\ni||#t\n")
    input = stringio("|hello m|y\tclear ||about\nworl|d of\ti||#t let| clear\n")
    output = stringio()

    convert([dictionary1, dictionary2], [input], output, None, True)

    expectedOutput = stringio()
    expectedOutput.write("0\t|S0 0:1 |# |#hello\t|S1 3:1 |# clear\n")
    expectedOutput.write("0\t|S0 1:1 |# m|#y\t|S1 4:1 |# |#|#about\n")
    expectedOutput.write("1\t|S0 2:1 |# worl|#d\t|S1 5:1 |# i|#|#t\n")
    expectedOutput.write("1\t|S0 3:1 |# of\t|S1 0:1 |# let|#\n")
    expectedOutput.write("1\t\t|S1 3:1 |# clear\n")
    for x in zip(output.getvalue().split('\n'), expectedOutput.getvalue().split('\n')):
        assert x[0] == x[1]

def test_nonExistingWord():
    dictionary1 = stringio("hello\nmy\nworld\nof\nnothing\n")
    input = stringio("hello my\nworld of nonexistent\n")
    output = stringio()

    with pytest.raises(Exception) as info:
        convert([dictionary1], [input], output, None, False)
    assert str(info.value) == "Token 'nonexistent' cannot be found in the dictionary for stream 0"
