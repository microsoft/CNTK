# file: runme.py

import example

print "ICONST  =", example.ICONST, "(should be 42)"
print "FCONST  =", example.FCONST, "(should be 2.1828)"
print "CCONST  =", example.CCONST, "(should be 'x')"
print "CCONST2 =", example.CCONST2, "(this should be on a new line)"
print "SCONST  =", example.SCONST, "(should be 'Hello World')"
print "SCONST2 =", example.SCONST2, "(should be '\"Hello World\"')"
print "EXPR    =", example.EXPR, "(should be 48.5484)"
print "iconst  =", example.iconst, "(should be 37)"
print "fconst  =", example.fconst, "(should be 3.14)"

try:
    print "EXTERN = ", example.EXTERN, "(Arg! This shouldn't print anything)"
except AttributeError:
    print "EXTERN isn't defined (good)"

try:
    print "FOO    = ", example.FOO, "(Arg! This shouldn't print anything)"
except AttributeError:
    print "FOO isn't defined (good)"
