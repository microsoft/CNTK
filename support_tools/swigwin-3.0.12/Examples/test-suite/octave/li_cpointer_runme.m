# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

li_cpointer


p = new_intp();
intp_assign(p,3);

if (intp_value(p) != 3)
    error
endif

delete_intp(p);

