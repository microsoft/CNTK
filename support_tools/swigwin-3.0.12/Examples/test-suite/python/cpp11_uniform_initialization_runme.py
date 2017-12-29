import cpp11_uniform_initialization

var1 = cpp11_uniform_initialization.cvar.var1
if var1.x != 5:
    raise RuntimeError
var2 = cpp11_uniform_initialization.cvar.var2
if var2.getX() != 2:
    raise RuntimeError

m = cpp11_uniform_initialization.MoreInit()
if m.charptr != None:
    raise RuntimeError, m.charptr
m.charptr = "hello sir"
if m.charptr != "hello sir":
    raise RuntimeError, m.charptr
if m.more1(m.vi) != 15:
    raise RuntimeError, m.vi
if m.more1([-1, 1, 2]) != 2:
    raise RuntimeError, m.vi
if m.more1() != 10:
    raise RuntimeError
