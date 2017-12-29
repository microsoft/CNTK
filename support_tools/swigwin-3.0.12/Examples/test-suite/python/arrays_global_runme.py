import arrays_global

arrays_global.cvar.array_i = arrays_global.cvar.array_const_i

from arrays_global import *

BeginString_FIX44a
cvar.BeginString_FIX44b
BeginString_FIX44c
cvar.BeginString_FIX44d
cvar.BeginString_FIX44d
cvar.BeginString_FIX44b = "12"'\0'"45"
cvar.BeginString_FIX44b
cvar.BeginString_FIX44d
cvar.BeginString_FIX44e
BeginString_FIX44f

test_a("hello", "hi", "chello", "chi")

test_b("1234567", "hi")
