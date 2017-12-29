import cpp11_function_objects
import sys


class Test1(cpp11_function_objects.Test):

    def __init__(self):
        cpp11_function_objects.Test.__init__(self)

    def __call__(self, a, b):
        self.value = a * b

t = cpp11_function_objects.Test()
if t.value != 0:
    raise RuntimeError(
        "Runtime cpp11_function_objects failed. t.value should be 0, but is " + str(t.value))

t(1, 2)  # adds numbers and sets value

if t.value != 3:
    raise RuntimeError(
        "Runtime cpp11_function_objects failed. t.value not changed - should be 3, but is " + str(t.value))

t2 = Test1()
a = cpp11_function_objects.testit1(t2, 4, 3)
if a != 12:
    raise RuntimeError(
        "Runtime cpp11_function_objects failed. t.value not changed - should be 12, but is " + str(a))
