exec("swigtest.start", -1);

foo_ptr = Foo_ptr_getPtr();
foo = Foo_ptr___deref__(foo_ptr);
checkequal(Foo_getVal(foo), 17, "Foo_getVal(p)");

foo_ptr = Foo_ptr_getConstPtr();
foo = Foo_ptr___deref__(foo_ptr);
checkequal(Foo_getVal(foo), 17, "Foo_getVal(p)");

exec("swigtest.quit", -1);
