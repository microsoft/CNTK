import preproc

if preproc.endif != 1:
    raise RuntimeError

if preproc.define != 1:
    raise RuntimeError

if preproc.defined != 1:
    raise RuntimeError

if 2 * preproc.one != preproc.two:
    raise RuntimeError

if preproc.methodX(99) != 199:
    raise RuntimeError
