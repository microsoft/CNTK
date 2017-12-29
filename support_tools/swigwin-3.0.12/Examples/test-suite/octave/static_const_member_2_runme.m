# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

static_const_member_2

c = Test_int();
try
    a = c.forward_field;
    a = c.current_profile;
    a = c.RightIndex;
    a = Test_int.backward_field;
    a = Test_int.LeftIndex;
    a = Test_int.cavity_flags;
catch
end_try_catch


if (Foo.BAZ.val != 2*Foo.BAR.val)
    error
endif

