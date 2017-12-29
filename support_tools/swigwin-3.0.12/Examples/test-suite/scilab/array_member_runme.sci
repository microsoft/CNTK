exec("swigtest.start", -1);

f = new_Foo();
Foo_data_set(f, [0:7]);
checkequal(Foo_data_get(f), [0:7], "Foo_data_get()");

Foo_text_set(f, "abcdefgh");
checkequal(Foo_text_get(f), "abcdefgh", "Foo_text_get()");
delete_Foo(f);

m = new_MyBuff();
MyBuff_x_set(m, [0:11]);
checkequal(MyBuff_x_get(m), [0:11], "MyBuff_x_get()");
delete_MyBuff(m);

exec("swigtest.quit", -1);
