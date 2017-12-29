# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

constructor_copy

f1 = Foo1(3);
f11 = Foo1(f1);


if (f1.x != f11.x)
    error
endif


f8 = Foo8();
try
    f81 = Foo8(f8);
    good = 0;
catch
    good = 1;
end_try_catch

if (!good)
    error
endif


bi = Bari(5);
bc = Bari(bi);

if (bi.x != bc.x)
    error
endif
    

bd = Bard(5);
try
    bc = Bard(bd);
    good = 0;
catch
    good = 1;
end_try_catch

if (!good)
    error
endif

