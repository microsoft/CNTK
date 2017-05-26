#!python
# 
# Syntax: python TimitGetFiles.py TIMIT_base_directory_path
#
# Simple script to take the list of valid TIMIT utterances, and turn them into
# the script files needed for HCopy and CNTK.
#
# This program reads the TimitSubjectList.txt file, which is included in the
# demo distribution.  It then creates three files:
#        HCopyTimit.scp - script that converts MFCC to audio features
#        CntkTimit.scp - list of files that are read by CNTK for training
#        CntkTimitOutput.scp - list of output files for CNTK (likelhihood scores)
# All these files can be edited by hand if you don't want to run this python
# script.

import os, sys

# Set up the base directory name.  This will be prepended to each file name
# so hcopy knows where to find the wave files.
if len(sys.argv) > 1:
        baseDir = sys.argv[1]
        if os.path.isdir(baseDir) == False:
                print "Can't find TIMIT base directory: " + baseDir
                sys.exit(1)
else:
        print "Syntax: " + sys.argv[0] + " TIMIT_base_directory_path"
        sys.exit(1)

if not baseDir.endswith('\\') and not baseDir.endswith('/'):
        baseDir += '/'

hcopyScript = 'HCopyTimit.scp'
cnScript = 'CntkTimit.scp'
cnOutputScript = 'CntkTimitOutput.scp'

hcopyScriptFp = open(hcopyScript, 'w')
cnScriptFp = open(cnScript, 'w')
cnOutputScriptFp = open(cnOutputScript, 'w')
fileCount = 0

fileList = 'TimitSubjectList.txt'
fileListFp = open(fileList)

if !hcopyScriptFp or !cnScriptFp or !cnOutputScriptFp or !fileListFp:
        print "Can't open the necessary output files."
        sys.exit(0)

for origFile in fileListFp:
        origFile = origFile.strip()
        fullFile = baseDir + origFile
        
        # Flatten the output structure.  Replace / with -
        outFile = origFile.replace('/', '-').replace('\\', '-')
        featFile = 'Features/train-' + outFile
        hcopyScriptFp.write(fullFile+'.nst ' + featFile+'.fbank_zda\n')
        cnScriptFp.write(featFile+'.fbank_zda\n')
        cnOutputScriptFp.write('Output/train-'+outFile+'.log\n')
        fileCount += 1
        if fileCount > 100000000:                # Debugging
                break

fileListFp.close()
hcopyScriptFp.close()
cnScriptFp.close()
cnOutputScriptFp.close()

