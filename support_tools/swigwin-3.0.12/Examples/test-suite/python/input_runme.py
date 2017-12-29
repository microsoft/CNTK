from input import *

f = Foo()
if f.foo(2) != 4:
    raise RuntimeError

if f.foo(None) != None:
    raise RuntimeError

if f.foo() != None:
    raise RuntimeError

if sfoo("Hello") != "Hello world":
    raise RuntimeError

if sfoo(None) != None:
    raise RuntimeError

if sfoo() != None:
    raise RuntimeError
