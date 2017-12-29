exec("swigtest.start", -1);

globalfunc();

f = new_Foo();
Foo_memberfunc(f);

Foo_staticmemberfunc();

v1 = vfunc1(f);
checkequal(SWIG_this(v1), SWIG_this(f), "vfunc1(f) <> f");

v2 = vfunc2(f);
checkequal(SWIG_this(v2), SWIG_this(f), "vfunc2(f) <> f");

v3 = vfunc3(v1);
checkequal(SWIG_this(v3), SWIG_this(f), "vfunc3(f) <> f");

Foo_memberfunc(v3);

exec("swigtest.quit", -1);

