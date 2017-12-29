# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

li_cmalloc

p = malloc_int();
free_int(p);

ok = 0;
try
    p = calloc_int(-1);
    free_int(p);
catch
    ok = 1;
end_try_catch

if (ok != 1)
    error
endif

