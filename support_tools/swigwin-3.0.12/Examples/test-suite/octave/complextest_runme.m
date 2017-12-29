# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

complextest

a = complex(-1,2);

if (complextest.Conj(a) != a.conjugate())
  error("bad complex mapping")
endif

if (complextest.Conjf(a) != a.conjugate())
  error("bad complex mapping")
endif


v = (complex(1,2), complex(2,3), complex(4,3), 1);

try
  complextest.Copy_h(v);
catch
end_try_catch
