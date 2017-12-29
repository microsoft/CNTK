# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

template_matrix
passVector([1,2,3]);
passMatrix({[1,2],[1,2,3]});
passCube({{[1,2],[1,2,3]},{[1,2],[1,2,3]}});


