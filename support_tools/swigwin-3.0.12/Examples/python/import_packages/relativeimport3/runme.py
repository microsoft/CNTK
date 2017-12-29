import sys
import os.path

# Test import of modules content from within __init__.py
testname = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
print "Testing " + testname + " - %module(package=...) with -relativeimport"

if sys.version_info < (3, 0):
    import py2.pkg2.bar
    print "  Finished importing py2.pkg2.bar"
else:
    import py3.pkg2.bar
    print "  Finished importing py3.pkg2.bar"
