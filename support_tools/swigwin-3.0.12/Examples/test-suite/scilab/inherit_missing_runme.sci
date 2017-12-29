exec("swigtest.start", -1);

a = new_Foo();
b = new_Bar();
c = new_Spam();

checkequal(do_blah(a), "Foo::blah", "do_blah(a)");
checkequal(do_blah(b), "Bar::blah", "do_blah(b)");
checkequal(do_blah(c), "Spam::blah", "do_blah(c)");

delete_Foo(a);
delete_Bar(b);
delete_Spam(c);

exec("swigtest.quit", -1);
