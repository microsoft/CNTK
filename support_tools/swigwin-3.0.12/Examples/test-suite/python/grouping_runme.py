import grouping

x = grouping.test1(42)
if x != 42:
    raise RuntimeError

grouping.test2(42)

x = grouping.do_unary(37, grouping.NEGATE)
if x != -37:
    raise RuntimeError

grouping.cvar.test3 = 42
grouping.test3 = 42
