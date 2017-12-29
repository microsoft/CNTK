exec("swigtest.start", -1);

// Test on NULL
null = getNull();
checkequal(SWIG_this(null), 0, "SWIG_this(null)");

null = SWIG_ptr(0);
checkequal(isNull(null), %T, "func(null)");

// Test on variable
expected_foo_addr = getFooAddress();
foo_addr = SWIG_this(pfoo_get());
checkequal(foo_addr, expected_foo_addr, "SWIG_this(pfoo_get())");

pfoo = SWIG_ptr(foo_addr);
checkequal(equalFooPointer(pfoo), %T, "equalFooPointer(pfoo)");

// Test conversion of mlist type pointers
stA = new_structA();
checkequal(typeof(stA), "_p_structA");
p = SWIG_ptr(stA);
checkequal(typeof(p), "pointer");

exec("swigtest.quit", -1);
