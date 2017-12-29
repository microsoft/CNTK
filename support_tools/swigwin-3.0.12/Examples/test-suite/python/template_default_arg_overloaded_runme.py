from template_default_arg_overloaded import *

def check(expected, got):
  if expected != got:
    raise RuntimeError("Expected: " + str(expected) + " got: " + str(got))


pl = PropertyList()
check(1, pl.setInt("int", 10))
check(1, pl.setInt("int", 10, False))

check(2, pl.set("int", pl))
check(2, pl.set("int", pl, False))

check(3, pl.setInt("int", 10, "int"))
check(3, pl.setInt("int", 10, "int", False))


pl = PropertyListGlobal()
check(1, pl.setIntGlobal("int", 10))
check(1, pl.setIntGlobal("int", 10, False))

check(2, pl.set("int", pl))
check(2, pl.set("int", pl, False))

check(3, pl.setIntGlobal("int", 10, "int"))
check(3, pl.setIntGlobal("int", 10, "int", False))


check(1, GoopIntGlobal(10))
check(1, GoopIntGlobal(10, True))

check(2, goopGlobal(3))
check(2, goopGlobal())

check(3, GoopIntGlobal("int", False))
check(3, GoopIntGlobal("int"))


check(1, GoopInt(10))
check(1, GoopInt(10, True))

check(2, goop(3))
check(2, goop())

check(3, GoopInt("int", False))
check(3, GoopInt("int"))
