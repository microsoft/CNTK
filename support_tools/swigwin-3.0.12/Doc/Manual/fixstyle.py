#!/usr/bin/python

# Replace the inline htmldoc stylesheet with the SWIG stylesheet

import sys
import string

filename = sys.argv[1]

data = open(filename).read()
open(filename+".bak","w").write(data)

swigstyle = "\n" + open("style.css").read()

lines = data.splitlines()
result = [ ]
skip = False
for s in lines:
    if not skip:
        result.append(s)
    if s == "<STYLE TYPE=\"text/css\"><!--":
        result.append(swigstyle)
        skip = True
    elif s == "--></STYLE>":
        result.append(s)
        skip = False

data = "\n".join(result)

open(filename,"w").write(data)
