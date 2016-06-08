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

    def __init__(self, dictionaryFiles, inputFiles, output, streamSeparator, comment):
        self.dictionaryFiles = dictionaryFiles
        self.inputFiles = inputFiles
        self.streamSeparator = streamSeparator
        self.output = output
        self.comment = comment

    def convert(self):
        dictionaries = self._createDictionaries()
        self._convertInputFiles(dictionaries)

    def _createDictionaries(self):
        dictionaries = []
        for dic in self.dictionaryFiles:
            dictionaries.append(self._createDictionary(dic))
        return dictionaries

    def _createDictionary(self, dictionaryFile):
        result = {}
        counter = 0
        for line in open(dictionaryFile):
            line = line.rstrip('\r\n').strip('\t ')
            result[line] = counter
            counter += 1
        return result

    def _convertInputFiles(self, dictionaries):
        for inputFile in self.inputFiles:
            self._convertInputFile(dictionaries, inputFile)

    def _convertInputFile(self, dictionaries, inputFile):
        sequenceId = 0
        for line in open(inputFile):
            line = line.rstrip('\r\n')
            streams = line.split("\t")
            if len(streams) != len(dictionaries):
                raise Exception("Number of dictionaries %(n) does no correspond to the number of streams in the line: %(line)" % (len(dictionaries), line))
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
                self.output.write("|S" + str(streamIndex) + " "+ str(value) + ":1")
                if self.comment:
                    self.output.write("|# " + token)
                self.output.write(self.streamSeparator)
            self.output.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transforms text file given  dictionaries into cntk text format.")
    parser.add_argument('--map', help='List of dictionaries,given in the same order as streams in the input files', required=True)
    parser.add_argument('--sep', help='Stream separator, default TAB', default="\t", required=False)
    parser.add_argument('--comment', help='Whether to annotate indexes with tokens. Default is false', choices=["True", "False"], default="False", required=False)
    parser.add_argument('--out', help='Name of the output file, stdout if not given', default="", required=False)
    parser.add_argument('inputFiles', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # cleaning dictionaryFiles from commas
    dictionaryFiles = "".join(str(x) for x in args.map).split(",")

    output = sys.stdout
    if args.out != "":
        output = open(output, "w")

    converter = Txt2CftConverter(dictionaryFiles, args.inputFiles, output, args.sep, args.comment == "True")
    converter.convert()
