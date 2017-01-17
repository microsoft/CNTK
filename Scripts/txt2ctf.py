#!/usr/bin/env python

# This script takes a list of dictionary files and a plain text file and converts this text input file to CNTK text format.
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
#    sed -e 's/^<\/s> //' -e 's/ <\/s>$//' < en.txt > en.txt1
#    sed -e 's/^<\/s> //' -e 's/ <\/s>$//' < fr.txt > fr.txt1
#    paste en.txt1 fr.txt1 | txt2ctf.py --map en.dict fr.dict > en-fr.ctf
#

import sys
import argparse

def convert(dictionaryStreams, inputs, output, annotated):
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
            _convertSequence(dictionaries, columns, sequenceId, output, annotated)
            sequenceId += 1

def _convertSequence(dictionaries, streams, sequenceId, output, annotated):
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
            if token not in dictionaries[streamIndex]:
                raise Exception("Token '{0}' cannot be found in the dictionary for stream {1}".format(token, streamIndex))
            value = dictionaries[streamIndex][token]
            output.write("\t|S" + str(streamIndex) + " "+ str(value) + ":1")
            if annotated:
                output.write(" |# " + token)
        output.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transforms text file given dictionaries into CNTK text format.")
    parser.add_argument('--map', help='List of dictionaries, given in the same order as streams in the input files',
        nargs="+", required=True)
    parser.add_argument('--annotated', help='Whether to annotate indices with tokens. Default is false',
        choices=["True", "False"], default="False", required=False)
    parser.add_argument('--output', help='Name of the output file, stdout if not given', default="", required=False)
    parser.add_argument('--input', help='Name of the inputs files, stdin if not given', default="", nargs="*", required=False)
    args = parser.parse_args()

    # creating inputs
    inputs = [sys.stdin]
    if len(args.input) != 0:
        inputs = [open(i) for i in args.input]

    # creating output
    output = sys.stdout
    if args.output != "":
        output = open(args.output, "w")

    convert([open(d) for d in args.map], inputs, output, args.annotated == "True")


#####################################################################################################
# Tests
#####################################################################################################

import StringIO
import pytest

def test_simpleSanityCheck():
    dictionary1 = StringIO.StringIO("hello\nmy\nworld\nof\nnothing\n")
    dictionary2 = StringIO.StringIO("let\nme\nbe\nclear\nabout\nit\n")
    input = StringIO.StringIO("hello my\tclear about\nworld of\tit let clear\n")
    output = StringIO.StringIO()

    convert([dictionary1, dictionary2], [input], output, False)

    expectedOutput = StringIO.StringIO()
    expectedOutput.write("0\t|S0 0:1\t|S1 3:1\n")
    expectedOutput.write("0\t|S0 1:1\t|S1 4:1\n")
    expectedOutput.write("1\t|S0 2:1\t|S1 5:1\n")
    expectedOutput.write("1\t|S0 3:1\t|S1 0:1\n")
    expectedOutput.write("1\t\t|S1 3:1\n")

    assert expectedOutput.getvalue() == output.getvalue()

def test_nonExistingWord():
    dictionary1 = StringIO.StringIO("hello\nmy\nworld\nof\nnothing\n")
    input = StringIO.StringIO("hello my\nworld of nonexistent\n")
    output = StringIO.StringIO()

    with pytest.raises(Exception) as info:
        convert([dictionary1], [input], output, False)
    assert info.value.message == "Token 'nonexistent' cannot be found in the dictionary for stream 0"
