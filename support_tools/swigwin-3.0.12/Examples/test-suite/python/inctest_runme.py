import inctest

error = 0
try:
    a = inctest.A()
except:
    print "didn't find A"
    print "therefore, I didn't include 'testdir/subdir1/hello.i'"
    error = 1
pass


try:
    b = inctest.B()
except:
    print "didn't find B"
    print "therefore, I didn't include 'testdir/subdir2/hello.i'"
    error = 1
pass

if error == 1:
    raise RuntimeError

# Check the import in subdirectory worked
if inctest.importtest1(5) != 15:
    print "import test 1 failed"
    raise RuntimeError

if inctest.importtest2("black") != "white":
    print "import test 2 failed"
    raise RuntimeError
