# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

template_ref_type

xr = template_ref_type.XC();
y  = template_ref_type.Y();
y.find(xr);
