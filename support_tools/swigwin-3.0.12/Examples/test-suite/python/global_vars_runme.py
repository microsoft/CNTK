import global_vars

global_vars.init()
b = global_vars.cvar.b
if b != "string b":
    raise RuntimeError("Unexpected string: " + b)
global_vars.cvar.b = "a string value"
b = global_vars.cvar.b
if b != "a string value":
    raise RuntimeError("Unexpected string: " + b)

x = global_vars.cvar.x
if x != 1234:
    raise RuntimeError("Unexpected x: " + str(x))
global_vars.cvar.x = 9876
x = global_vars.cvar.x
if x != 9876:
    raise RuntimeError("Unexpected string: " + str(x))

fail = True
try:
    global_vars.cvar.notexist = "something"
except AttributeError, e:
    fail = False
if fail:
    raise RuntimeError("AttributeError should have been thrown")

fail = True
try:
    g = global_vars.cvar.notexist
except AttributeError, e:
    fail = False
if fail:
    raise RuntimeError("AttributeError should have been thrown")
