from using_composition import *

f = FooBar()
if f.blah(3) != 3:
    raise RuntimeError, "FooBar::blah(int)"

if f.blah(3.5) != 3.5:
    raise RuntimeError, "FooBar::blah(double)"

if f.blah("hello") != "hello":
    raise RuntimeError, "FooBar::blah(char *)"


f = FooBar2()
if f.blah(3) != 3:
    raise RuntimeError, "FooBar2::blah(int)"

if f.blah(3.5) != 3.5:
    raise RuntimeError, "FooBar2::blah(double)"

if f.blah("hello") != "hello":
    raise RuntimeError, "FooBar2::blah(char *)"


f = FooBar3()
if f.blah(3) != 3:
    raise RuntimeError, "FooBar3::blah(int)"

if f.blah(3.5) != 3.5:
    raise RuntimeError, "FooBar3::blah(double)"

if f.blah("hello") != "hello":
    raise RuntimeError, "FooBar3::blah(char *)"
