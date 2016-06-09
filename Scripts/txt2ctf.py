#!/usr/bin/env python

# This script takes a list of dictionaries and the input file with text and converts the text file to cntk text format
#
# Input file should be in the following form:
#    text1 TAB text2 TAB ... TAB textN
# where each line represents one sequence across all input streams (N)
# Each text consists of space separated tokens (samples)
#
# Dictionaries are text files that are specified for all streams, so the #dictionaries = #columns in the input file.
# A dictionary contain a single token per line. The line number becomes the numeric index of the token.

# The converter takes the input file and create the corresponding cntk format file using the numeric indexes of tokens.

import sys
import argparse

class Txt2CftConverter:
    """Class that converts tab separated sequences into cntk text format
       Each line in the input file should be of form <text1 TAB ... TAB textN>, where N is the number of streams
       Each text is a list of space separated tokens(samples)
       Each token for a stream should be inside the corresponding dictionary file, a token per line, so the line number of the token becomes
       the numeric index written into the cntk text format output file"""

    def __init__(self, dictionaries, inputs, output, streamSeparator, comment):
        self.dictionaries = dictionaries
        self.inputs = inputs
        self.streamSeparator = streamSeparator
        self.output = output
        self.comment = comment

    def convert(self):
        dictionaries = self._createDictionaries()
        self._convertInputs(dictionaries)

    def _createDictionaries(self):
        dictionaries = []
        for dic in self.dictionaries:
            dictionaries.append(self._createDictionary(dic))
        return dictionaries

    def _createDictionary(self, dictionary):
        result = {}
        counter = 0
        for line in dictionary:
            line = line.rstrip('\r\n').strip('\t ')
            result[line] = counter
            counter += 1
        return result

    def _convertInputs(self, dictionaries):
        if len(self.inputs) == 0:
            return self._convertInput(dictionaries, sys.stdin)
        for input in self.inputs:
            self._convertInput(dictionaries, input)

    def _convertInput(self, dictionaries, input):
        sequenceId = 0
        for line in input:
            line = line.rstrip('\r\n')
            streams = line.split("\t")
            if len(streams) != len(dictionaries):
                raise Exception("Number of dictionaries {0} does no correspond to the number of streams in the line: {1}".format(len(dictionaries), line))
            self._convertStreams(dictionaries, streams, sequenceId)
            sequenceId += 1

    def _convertStreams(self, dictionaries, streams, sequenceId):
        tokenizedStreams = []
        maxLen = 0
        for index, stream in enumerate(streams):
            streamTokens = stream.strip(' ').split(' ')
            streamTokens = [t for t in streamTokens if t != ""]
            tokenizedStreams.append(streamTokens)
            if len(streamTokens) > maxLen:
                maxLen = len(streamTokens)

        # writing to the output file
        for sampleIndex in range(maxLen):
            self.output.write(str(sequenceId) + "\t")
            for streamIndex in range(len(tokenizedStreams)):
                if len(tokenizedStreams[streamIndex]) <= sampleIndex:
                    self.output.write(self.streamSeparator)
                    continue
                token = tokenizedStreams[streamIndex][sampleIndex]
                value = dictionaries[streamIndex][token]
                self.output.write(self.streamSeparator)
                self.output.write("|S" + str(streamIndex) + " "+ str(value) + ":1")
                if self.comment:
                    self.output.write("|# " + token)
            self.output.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transforms text file given  dictionaries into cntk text format.")
    parser.add_argument('--map', help='List of dictionaries,given in the same order as streams in the input files', required=True)
    parser.add_argument('--sep', help='Stream separator, default TAB', default="\t", required=False)
    parser.add_argument('--comment', help='Whether to annotate indexes with tokens. Default is false', choices=["True", "False"], default="False", required=False)
    parser.add_argument('--output', help='Name of the output file, stdout if not given', default="", required=False)
    parser.add_argument('--input', help='Name of the inputs files, stdin if not given', default="", required=False)
    args = parser.parse_args()

    # creating dictionaries
    dictionaryFiles = "".join(str(x) for x in args.map).split(",")    
    dictionaries = open(d) for d in dictionaryFiles
    
    # creating inputs
    inputs = [sys.stdin]
    if args.input != "":
        inputFiles = "".join(str(x) for x in args.input).split(",")
        inputs = open(i) for i in inputFiles

    # creating outputs
    output = sys.stdout
    if args.output != "":
        output = open(args.output, "w")

    converter = Txt2CftConverter(dictionaries, inputs, output, args.sep, args.comment == "True")
    converter.convert()

# Test
import StringIO

def test_sanityCheck():
    dictionary1 = StringIO.StringIO()
    dictionary1.write("hello\nmy\nworld\nof\nnothing\n")
    
    dictionary2 = StringIO.StringIO()
    dictionary2.write("let\nme\nbe\nclear\nabout\nit\n")
    
    input = StringIO.StringIO()
    input.write("hello my\tclear about\nworld of\tit let clear\n")

    output = StringIO.StringIO()
    converter = Txt2CftConverter([dictionary1, dictionary2], [input], output, "\t", False)
    
    expectedOutput = StringIO.StringIO()
    expectedOutput.write("0\t|S0 0:1\t|S1 3:1\n")
    expectedOutput.write("0\t|S0 1:1\t|S1 4:1\n")
    expectedOutput.write("1\t|S0 2:1\t|S1 5:1\n")
    expectedOutput.write("1\t|S0 3:1\t|S1 0:1\n")
    expectedOutput.write("1\t\t|S1 3:1")
    
    assert expectedOutput.content() == output.content()
