compactdefaultargs

defaults1 = Defaults1(1000);
defaults1 = Defaults1();

if (defaults1.ret(10.0) != 10.0)
  error
endif

if (defaults1.ret() != -1.0)
  error
endif

defaults2 = Defaults2(1000);
defaults2 = Defaults2();

if (defaults2.ret(10.0) != 10.0)
  error
endif

if (defaults2.ret() != -1.0)
  error
endif

