import abstract_access

d = abstract_access.D()
if d.do_x() != 1:
    raise RuntimeError
