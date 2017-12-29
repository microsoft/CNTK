from using_inherit import *

b = Bar()
if b.test(3) != 3:
    raise RuntimeError, "Bar::test(int)"

if b.test(3.5) != 3.5:
    raise RuntimeError, "Bar::test(double)"


b = Bar2()
if b.test(3) != 6:
    raise RuntimeError, "Bar2::test(int)"

if b.test(3.5) != 7.0:
    raise RuntimeError, "Bar2::test(double)"


b = Bar3()
if b.test(3) != 6:
    raise RuntimeError, "Bar3::test(int)"

if b.test(3.5) != 7.0:
    raise RuntimeError, "Bar3::test(double)"


b = Bar4()
if b.test(3) != 6:
    raise RuntimeError, "Bar4::test(int)"

if b.test(3.5) != 7.0:
    raise RuntimeError, "Bar4::test(double)"


b = Fred1()
if b.test(3) != 3:
    raise RuntimeError, "Fred1::test(int)"

if b.test(3.5) != 7.0:
    raise RuntimeError, "Fred1::test(double)"


b = Fred2()
if b.test(3) != 3:
    raise RuntimeError, "Fred2::test(int)"

if b.test(3.5) != 7.0:
    raise RuntimeError, "Fred2::test(double)"
