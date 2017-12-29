import preproc_defined

if preproc_defined.call_checking() != 1:
    raise RuntimeError

d = preproc_defined.Defined()
d.defined = 10

preproc_defined.thing(10)
preproc_defined.stuff(10)
preproc_defined.bumpf(10)
