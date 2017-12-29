# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

friends

a = friends.A(2);

if (friends.get_val1(a) != 2)
  error
endif
if (friends.get_val2(a) != 4)
  error
endif
if (friends.get_val3(a) != 6)
  error
endif

				# nice overload working fine
if (friends.get_val1(1,2,3) != 1)
  error
endif

b = friends.B(3);

				# David's case
if (friends.mix(a,b) != 5)
  error
endif

di = friends.D_d(2);
dd = friends.D_d(3.3);

				# incredible template overloading working just fine
if (friends.get_val1(di) != 2)
  error
endif
if (friends.get_val1(dd) != 3.3)
  error
endif

friends.set(di, 4);
friends.set(dd, 1.3);

if (friends.get_val1(di) != 4)
  error
endif
if (friends.get_val1(dd) != 1.3)
  error
endif
