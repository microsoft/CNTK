#/usr/bin/env python

import sys
import string
import os

def usage():
    print ('convert_scp_to_cntk.py -in [fea_list] [label_list]')

def createDir(d):
    if not os.path.isdir(d):
        os.makedirs(d)

if len(sys.argv) != 4:
    usage()
else:
    fr = open(sys.argv[2], 'r')
    lines = [x.rstrip() for x in fr]
    fr.close()
    
    fr = open (sys.argv[3], 'r')
    lablines = [x.rstrip() for x in fr]
    fr.close()
    frnum=0
    for line in lines:
        fr = open (lablines[frnum], 'r')
        labs = [x.rstrip() for x in fr]
        fr.close()
        lenLab = len(labs)-1
        print (line + "[0,"+str(lenLab)+"]")
        frnum = frnum + 1
