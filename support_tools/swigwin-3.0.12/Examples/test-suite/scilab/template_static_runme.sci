exec("swigtest.start", -1);

checkequal(foo_i_test_get(), 0, "foo_i_test_get() test fails.");
checkequal(foo_d_test_get(), 0, "foo_i_test_get() test fails.");

checkequal(Foo_bar_double(0), 1, "Foo_bar_double() test fails");

exec("swigtest.quit", -1);
