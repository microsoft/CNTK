from using_extend import *

f = FooBar()
if f.blah(3) != 3:
    raise RuntimeError, "blah(int)"

if f.blah(3.5) != 3.5:
    raise RuntimeError, "blah(double)"

if f.blah("hello") != "hello":
    raise RuntimeError, "blah(char *)"

if f.blah(3, 4) != 7:
    raise RuntimeError, "blah(int,int)"

if f.blah(3.5, 7.5) != (3.5 + 7.5):
    raise RuntimeError, "blah(double,double)"


if f.duh(3) != 3:
    raise RuntimeError, "duh(int)"
