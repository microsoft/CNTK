exec("swigtest.start", -1);

checkequal(test("Hello"), "Hello", "test(""Hello"")");

f = new_Foo("Greetings");
checkequal(Foo_str_get(f), "Greetings", "new_Foo(""Greetings"")");

checkequal(Foo_test(f, "Hello"), "Hello", "Foo_test(f)");
delete_Foo(f);

checkequal(Foo_statictest("Hello", 1), "Hello", "Foo_statictest(""Hello"", 1)");

checkequal(test_def("Hello", 1), "Hello", "test_def(""Hello"", 1)");

checkequal(test_def("Hello"), "Hello", "test_def(""Hello"")");

exec("swigtest.quit", -1);
