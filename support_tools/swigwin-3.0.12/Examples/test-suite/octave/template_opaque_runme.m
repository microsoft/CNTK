# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

template_opaque

v = template_opaque.OpaqueVectorType(10);

template_opaque.FillVector(v);


