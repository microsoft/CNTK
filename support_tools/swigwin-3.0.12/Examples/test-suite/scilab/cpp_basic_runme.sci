exec("swigtest.start", -1);

f = new_Foo(4);
checkequal(Foo_num_get(f), 4, "Foo_num_get(f)");
Foo_num_set(f, -17);
checkequal(Foo_num_get(f), -17, "Foo_num_get(f)");

b = new_Bar();
Bar_fptr_set(b, f);

fptr = Bar_fptr_get(b);
checkequal(Foo_num_get(fptr), -17, "Foo_num_get(ftr)");

checkequal(Bar_test(b, -3, fptr), -5, "Bar_test(b, -3, fptr)");

fref = Bar_fref_get(b);
checkequal(Foo_num_get(fref), -4, "Foo_num_get(fref)");

checkequal(Bar_test(b, 12, fref), 23, "Bar_test(b, 12, fref)");

f2 = new_Foo(23);
Bar_fref_set(b, f2);

fref = Bar_fref_get(b);
checkequal(Foo_num_get(fref), 23, "Foo_num_get(fref)");

fval = Bar_fval_get(b);
checkequal(Bar_test(b, 3, fval), 33, "Bar_test(b, 3, fval)");

Bar_fval_set(b, new_Foo(-15));

fval = Bar_fval_get(b);
checkequal(Foo_num_get(fval), -15, "Foo_num_get(fval)");
checkequal(Bar_test(b, 3, fval), -27, "Bar_test(b, 3, fval)");

f3 = Bar_testFoo(b, 12, fref);
checkequal(Foo_num_get(f3), 32, "Foo_num_get(f3)");


// Test globals
f4 = new_Foo(6);
Bar_global_fptr_set(f4);
checkequal(Foo_num_get(Bar_global_fptr_get()), 6, "Foo_num_get(Bar_global_fptr_get())");

checkequal(Foo_num_get(Bar_global_fref_get()), 23, "Foo_num_get(Bar_global_fref_get())");

checkequal(Foo_num_get(Bar_global_fval_get()), 3, "Foo_num_get(Bar_global_fval_get())");


// Test member function pointers
func1_ptr = get_func1_ptr();
func2_ptr = get_func2_ptr();

Foo_num_set(f, 4);
checkequal(Foo_func1(f, 2), 16, "Foo_func1(f, 2)");
checkequal(Foo_func2(f, 2), -8, "Foo_func2(f, 2)");

Foo_func_ptr_set(f, func1_ptr);
checkequal(test_func_ptr(f, 2), 16, "Foo_test_func_ptr(f, 2)");

Foo_func_ptr_set(f, func2_ptr);
checkequal(test_func_ptr(f, 2), -8, "Foo_test_func_ptr(f, 2)");

exec("swigtest.quit", -1);
