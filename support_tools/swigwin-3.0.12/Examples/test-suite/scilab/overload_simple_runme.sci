exec("swigtest.start", -1);

f = new_Foo();
b = new_Bar();
v = malloc_void(32);

// Functions

checkequal(foo(int32(3)), "foo:int", "foo(int)");
checkequal(foo(3), "foo:double", "foo(double)");
checkequal(foo("hello"), "foo:char *", "foo(char* )");
checkequal(foo(f), "foo:Foo *", "foo(Foo *)");
checkequal(foo(b), "foo:Bar *", "foo(Bar *)");
checkequal(foo(v), "foo:void *", "foo(void *)");

// Class methods

s = new_Spam();
checkequal(Spam_foo(s, int32(3)), "foo:int", "Spam::foo(int)");
checkequal(Spam_foo(s, 3), "foo:double", "Spam::foo(double)");
checkequal(Spam_foo(s, "hello"), "foo:char *", "Spam::foo(char *)");
checkequal(Spam_foo(s, f), "foo:Foo *", "Spam::foo(Foo *)");
checkequal(Spam_foo(s, b), "foo:Bar *", "Spam::foo(Bar *)");
checkequal(Spam_foo(s, v), "foo:void *", "Spam::foo(void *)");
delete_Spam(s);

// Static class methods

checkequal(Spam_bar(int32(3)), "bar:int", "Spam::bar(int)");
checkequal(Spam_bar(3.1), "bar:double", "Spam::bar(double)");
checkequal(Spam_bar("hello"), "bar:char *", "Spam::bar(char *)");
checkequal(Spam_bar(f), "bar:Foo *", "Spam::bar(Foo *)");
checkequal(Spam_bar(b), "bar:Bar *", "Spam::bar(Bar *)");
checkequal(Spam_bar(v), "bar:void *", "Spam::bar(void *)");

// Constructors

s = new_Spam();
checkequal(Spam_type_get(s), "none", "Spam::Spam()");
delete_Spam(s);

s = new_Spam(int32(3));
checkequal(Spam_type_get(s), "int", "Spam::Spam(int)");
delete_Spam(s);

s = new_Spam(3.1);
checkequal(Spam_type_get(s), "double", "Spam::Spam(double)");
delete_Spam(s);

s = new_Spam("hello");
checkequal(Spam_type_get(s), "char *", "Spam::Spam(char *)");
delete_Spam(s);

s = new_Spam(f);
checkequal(Spam_type_get(s), "Foo *", "Spam::Spam(Foo *)");
delete_Spam(s);
    
s = new_Spam(b);
checkequal(Spam_type_get(s), "Bar *", "Spam::Spam(Bar *)");
delete_Spam(s);

s = new_Spam(v);
checkequal(Spam_type_get(s), "void *", "Spam::Spam(void *)");
delete_Spam(s);

delete_Foo(f);
delete_Bar(b);
free_void(v);

a = new_ClassA();
b = ClassA_method1(a, 1);
delete_ClassA(a);


exec("swigtest.quit", -1);
