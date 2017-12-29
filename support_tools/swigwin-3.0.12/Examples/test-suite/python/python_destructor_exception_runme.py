import python_destructor_exception
from StringIO import StringIO
import sys

def error_function():
    python_destructor_exception.ClassWithThrowingDestructor().GetBlah()

def runtest():
    attributeErrorOccurred = False
    try:
        error_function()
    except AttributeError, e:
        attributeErrorOccurred = True
    return attributeErrorOccurred

def test1():
    stderr_saved = sys.stderr
    buffer = StringIO()
    attributeErrorOccurred = False
    try:
        # Suppress stderr while making this call to suppress the output shown by PyErr_WriteUnraisable
        sys.stderr = buffer

        attributeErrorOccurred = runtest()
    finally:
        sys.stderr.flush()
        sys.stderr = stderr_saved

    if not attributeErrorOccurred:
        raise RuntimeError("attributeErrorOccurred failed")
    if not buffer.getvalue().count("I am the ClassWithThrowingDestructor dtor doing bad things") >= 1:
        raise RuntimeError("ClassWithThrowingDestructor dtor doing bad things failed")

class VectorHolder(object):
    def __init__(self, v):
        self.v = v
    def gen(self):
        for e in self.v:
            yield e

# See issue #559, #560, #573 - In Python 3.5, test2() call to the generator 'gen' was
# resulting in the following (not for -builtin where there is no call to SWIG_Python_CallFunctor
# as SwigPyObject_dealloc is not used):
#
# StopIteration
#
# During handling of the above exception, another exception occurred:
# ...
# SystemError: <built-in function delete_VectorInt> returned a result with an error set

def addup():
    sum = 0
    for i in VectorHolder(python_destructor_exception.VectorInt([1, 2, 3])).gen():
        sum = sum + i
    return sum

def test2():
    sum = addup()

    if sum != 6:
        raise RuntimeError("Sum is incorrect")

# These two tests are different are two different ways to recreate essentially the same problem
# reported by Python 3.5 that an exception was already set when destroying a wrapped object
test1()
test2()
