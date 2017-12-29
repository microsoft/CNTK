from char_binary import *

t = Test()
if t.strlen('hile') != 4:
    print t.strlen('hile')
    raise RuntimeError, "bad multi-arg typemap"
if t.ustrlen('hile') != 4:
    print t.ustrlen('hile')
    raise RuntimeError, "bad multi-arg typemap"

if t.strlen('hil\0') != 4:
    raise RuntimeError, "bad multi-arg typemap"
if t.ustrlen('hil\0') != 4:
    raise RuntimeError, "bad multi-arg typemap"

#
# creating a raw char*
#
pc = new_pchar(5)
pchar_setitem(pc, 0, 'h')
pchar_setitem(pc, 1, 'o')
pchar_setitem(pc, 2, 'l')
pchar_setitem(pc, 3, 'a')
pchar_setitem(pc, 4, 0)


if t.strlen(pc) != 4:
    raise RuntimeError, "bad multi-arg typemap"
if t.ustrlen(pc) != 4:
    raise RuntimeError, "bad multi-arg typemap"

cvar.var_pchar = pc
if cvar.var_pchar != "hola":
    print cvar.var_pchar
    raise RuntimeError, "bad pointer case"

cvar.var_namet = pc
# if cvar.var_namet != "hola\0":
if cvar.var_namet != "hola":
    raise RuntimeError, "bad pointer case"

delete_pchar(pc)
