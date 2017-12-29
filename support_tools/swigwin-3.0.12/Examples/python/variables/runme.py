# file: runme.py

import example

# Try to set the values of some global variables

example.cvar.ivar = 42
example.cvar.svar = -31000
example.cvar.lvar = 65537
example.cvar.uivar = 123456
example.cvar.usvar = 61000
example.cvar.ulvar = 654321
example.cvar.scvar = -13
example.cvar.ucvar = 251
example.cvar.cvar = "S"
example.cvar.fvar = 3.14159
example.cvar.dvar = 2.1828
example.cvar.strvar = "Hello World"
example.cvar.iptrvar = example.new_int(37)
example.cvar.ptptr = example.new_Point(37, 42)
example.cvar.name = "Bill"

# Now print out the values of the variables

print "Variables (values printed from Python)"

print "ivar      =", example.cvar.ivar
print "svar      =", example.cvar.svar
print "lvar      =", example.cvar.lvar
print "uivar     =", example.cvar.uivar
print "usvar     =", example.cvar.usvar
print "ulvar     =", example.cvar.ulvar
print "scvar     =", example.cvar.scvar
print "ucvar     =", example.cvar.ucvar
print "fvar      =", example.cvar.fvar
print "dvar      =", example.cvar.dvar
print "cvar      =", example.cvar.cvar
print "strvar    =", example.cvar.strvar
print "cstrvar   =", example.cvar.cstrvar
print "iptrvar   =", example.cvar.iptrvar
print "name      =", example.cvar.name
print "ptptr     =", example.cvar.ptptr, example.Point_print(example.cvar.ptptr)
print "pt        =", example.cvar.pt, example.Point_print(example.cvar.pt)

print "\nVariables (values printed from C)"

example.print_vars()

print "\nNow I'm going to try and modify some read only variables"

print "     Tring to set 'path'"
try:
    example.cvar.path = "Whoa!"
    print "Hey, what's going on?!?! This shouldn't work"
except:
    print "Good."

print "     Trying to set 'status'"
try:
    example.cvar.status = 0
    print "Hey, what's going on?!?! This shouldn't work"
except:
    print "Good."


print "\nI'm going to try and update a structure variable.\n"

example.cvar.pt = example.cvar.ptptr

print "The new value is"
example.pt_print()
print "You should see the value", example.Point_print(example.cvar.ptptr)
