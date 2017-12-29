# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

enum_template

if (enum_template.MakeETest() != 1)
  error
endif

enum_template.TakeETest(0);
try
  a=enum_template.TakeETest(0);
  error
catch
end_try_catch
  
