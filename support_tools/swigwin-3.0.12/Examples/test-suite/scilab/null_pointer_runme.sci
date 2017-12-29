exec("swigtest.start", -1);

p = getnull();
checkequal(SWIG_this(p), 0, "SWIG_this(p)");
checkequal(func(p), %T, "func(p)");

exec("swigtest.quit", -1);
