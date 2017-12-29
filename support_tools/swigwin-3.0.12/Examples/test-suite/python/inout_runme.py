import inout

a = inout.AddOne1(1)
if a != 2:
    raise RuntimeError

a = inout.AddOne3(1, 1, 1)
if a != [2, 2, 2]:
    raise RuntimeError

a = inout.AddOne1p((1, 1))
if a != (2, 2):
    raise RuntimeError

a = inout.AddOne2p((1, 1), 1)
if a != [(2, 2), 2]:
    raise RuntimeError

a = inout.AddOne3p(1, (1, 1), 1)
if a != [2, (2, 2), 2]:
    raise RuntimeError
