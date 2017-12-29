from argcargvtest import *

largs = ['hi', 'hola', 'hello']
if mainc(largs) != 3:
    raise RuntimeError("bad main typemap")

targs = ('hi', 'hola')
if mainv(targs, 1) != 'hola':
    print(mainv(targs, 1))
    raise RuntimeError("bad main typemap")

targs = ('hi', 'hola')
if mainv(targs, 1) != 'hola':
    raise RuntimeError("bad main typemap")

try:
    error = 0
    mainv('hello', 1)
    error = 1
except TypeError:
    pass
if error:
    raise RuntimeError("bad main typemap")


initializeApp(largs)
