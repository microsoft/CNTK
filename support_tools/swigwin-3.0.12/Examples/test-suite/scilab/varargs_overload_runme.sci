exec("swigtest.start", -1);

checkequal(vararg_over1("Hello"), "Hello", "vararg_over1(""Hello"")");

checkequal(vararg_over1(2), "2", "vararg_over1(2)");

checkequal(vararg_over2("Hello"), "Hello", "vararg_over1(""Hello"")");

checkequal(vararg_over2(2, 2.2), "2 2.2", "vararg_over2(2, 2.2)")

checkequal(vararg_over3("Hello"), "Hello", "vararg_over3(""Hello"")");

checkequal(vararg_over3(2, 2.2, "hey"), "2 2.2 hey", "vararg_over3(2, 2.2, ""hey"")");

checkequal(vararg_over4("Hello"), "Hello", "vararg_over4(""Hello"")");

checkequal(vararg_over4(123), "123", "vararg_over4(123)");

checkequal(vararg_over4("Hello", 123), "Hello", "vararg_over4(""Hello"", 123)");

exec("swigtest.quit", -1);
