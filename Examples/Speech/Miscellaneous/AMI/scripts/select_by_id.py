#/usr/bin/env python

import sys
import string
import os

def usage():
    print ('MapLabels.py -in [list file] -out_dir [out dir] -mapping [mapping file]')

def createDir(d):
    if not os.path.isdir(d):
        os.makedirs(d)

if 0: #len(sys.argv) != 7:
    usage()
else:
    fr = open(sys.argv[2], 'r')
    lines = [x.rstrip() for x in fr]
    fr.close()
    
    my_dict = {}
    for line in lines:
        linenew = line.split('.')
        tmp = linenew[0].split('/')
        my_dict[tmp[-1]] = line

    fr = open (sys.argv[3], 'r')
    lablines = [x.rstrip() for x in fr]
    fr.close()
    
    frnum=0
    for line in lablines:
        linenew = line.split('.')
        tmp = linenew[0].split('/')
        print (tmp[-1] + "=" + my_dict[tmp[-1]])

