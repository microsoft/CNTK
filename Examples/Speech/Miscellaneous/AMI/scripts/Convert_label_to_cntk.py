#/usr/bin/env python

import sys
import string
import os

def usage():
    print ('Convert_label_to_cntk.py -in [fea_list file] [label_list_file] delay_number')

def createDir(d):
    if not os.path.isdir(d):
        os.makedirs(d)

if len(sys.argv) != 5:
    usage()
else:
    fr = open(sys.argv[2], 'r')
    lines = [x.rstrip() for x in fr]
    fr.close()
    
    fr = open (sys.argv[3], 'r')
    delay = int(sys.argv[4])
    lablines = [x.rstrip() for x in fr]
    fr.close()
    print ("#!MLF!#")
    frnum=0
    for line in lines:
        linenew = line.split('.')
        tmp = linenew[0].split('/')
        print ("\""+tmp[-1]+"\"")
        fr = open (lablines[frnum], 'r')
        labs = [x.rstrip() for x in fr]
        fr.close()
        i = 0
        for lab in labs:
            j = i+1
            k = i-delay
            if i-delay < 0:
                k = 0
            print (i, j, labs[k], labs[k])
            i = i+1
        print (".")
        frnum = frnum + 1

