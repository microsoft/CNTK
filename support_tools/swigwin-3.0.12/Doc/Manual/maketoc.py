#!/usr/bin/env python

import sys
import os
chs = open("chapters").readlines()

f = open("Contents.html","w")
print >>f, """
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN" "http://www.w3.org/TR/html4/loose.dtd">
<HTML>
<HEAD>
<TITLE>SWIG Users Manual</TITLE>
<meta http-equiv="content-type" content="text/html; charset=UTF-8">
</HEAD>
<BODY BGCOLOR="#ffffff">

<H1><a name="Contents"></a>SWIG Users Manual</H1>

<p>
"""

f.close()

num = 1

for c in chs:
    c = c.strip()
    print "Processing %s" % c
    if c:
        os.system("python makechap.py %s %d >> Contents.html" % (c,num))
    num += 1
    
f = open("Contents.html","a")
print >>f, """
</BODY>
</HTML>
"""


