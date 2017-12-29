# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

arrays_global

arrays_global.cvar.array_i = arrays_global.cvar.array_const_i;

cvar.BeginString_FIX44a;
cvar.BeginString_FIX44b;
cvar.BeginString_FIX44c;
cvar.BeginString_FIX44d;
cvar.BeginString_FIX44d;
cvar.BeginString_FIX44b = strcat("12","\0","45");
cvar.BeginString_FIX44b;
cvar.BeginString_FIX44d;
cvar.BeginString_FIX44e;
cvar.BeginString_FIX44f;

test_a("hello","hi","chello","chi");

test_b("1234567","hi");

