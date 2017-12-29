import exception_classname

a = exception_classname.Exception()
if a.testfunc() != 42:
    raise RuntimeError("Not 42!")
