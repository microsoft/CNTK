#!/usr/bin/python

# Convert ASCII text into 1-hot vectors in UCI (space delimited) format
#
# AsciiTo1HotUCI.py <inputfile> <outputfile>
#
# Python string.printable (upper+lower+numbers+punctuation) contains 100 items
# '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r\x0b\x0c'
# note that the last few won't behave well in a text data file, so we will rewrite the corresponding data labels
import sys, os, string;

def __main__():
#
#    infilename = sys.argv[1];
#    outfilename = sys.argv[2];

    infilename = "/Users/hojohnl/Source/opencv/README.md.txt";

    infile = open(infilename, 'rU');
#    outfile = open(outfilename, 'wU');
    outfile = sys.stdout;



#   Build a dictionary of the symbols (characters) and associated vector component and label

    fwdtable = {};  # find the vector component index for this character
    revtable = {};  # find the vector label for this component index
    i = 0;
    for ch in string.printable:
        fwdtable[ch] = i;
        revtable[i] = str(ch) + "   ";  # make all labels 4 char wide for human readability
        i += 1;

# fix up special cases so we can see reasonable labels in our 1-hot input files
    revtable[94] = "<SP>"; # 0x20
    revtable[95] = "<HT>"; # 0x09    
    revtable[96] = "<LF>"  # 0x0a
    revtable[97] = "<CR>"  # 0x0d
    revtable[98] = "<VT>"  # 0x0b
    revtable[99] = "<FF>"; # 0x0c

    for i in range(100):
        print 'value %d, key %s' % (i, revtable[i]);
            
    for line in infile:
        printables = filter(lambda x: x in string.printable, line);

        for ch in printables:
            vector_index = fwdtable[ch];
            vector_label = revtable[vector_index];
            vector_text_list = list();

            vector_text_list.append(vector_label + " ");
            for i in range(100):
                vector_text_list.append('1 ' if (i == vector_index) else '0 ');
            vector_text_string = ''.join(vector_text_list);
            outfile.write(vector_text_string + '\n');

__main__();
            
