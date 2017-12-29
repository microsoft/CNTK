exec("swigtest.start", -1);

f = new_Foo();
b = new_Bar();

checkequal(spam(f), 1, "spam(f)");
checkequal(spam(b), 2, "spam(b)");

delete_Foo(f);
delete_Bar(b);

exec("swigtest.quit", -1);

