from python_append import *


def is_new_style_class(cls):
    return hasattr(cls, "__class__")

# test not relevant for -builtin
if is_python_builtin():
    exit(0)

t = Test()
t.func()
if is_new_style_class(Test):
    t.static_func()
else:
    Test_static_func()

if grabpath() != os.path.dirname(mypath):
    raise RuntimeError("grabpath failed")

if grabstaticpath() != os.path.basename(mypath):
    raise RuntimeError("grabstaticpath failed")

clearstaticpath()
if grabstaticpath() != None:
    raise RuntimeError("Resetting staticfuncpath failed")
Test.static_func()
if grabstaticpath() != os.path.basename(mypath):
    raise RuntimeError("grabstaticpath failed")
