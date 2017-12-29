# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

li_implicit
b = B();
ai = A(1);
ad = A(2.0);
ab = A(b);

ai, get(ai);
ad, get(ad);
ab, get(ab);

if (get(ai) != get(1))
  error("bad implicit type")
endif
if (get(ad) != get(2.0))
  error("bad implicit type")
endif
if (get(ab) != get(b))
  error("bad implicit type")
endif

