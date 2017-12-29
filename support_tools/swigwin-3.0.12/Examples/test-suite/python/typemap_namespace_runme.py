from typemap_namespace import *

if test1("hello") != "hello":
    raise RuntimeError

if test2("hello") != "hello":
    raise RuntimeError
