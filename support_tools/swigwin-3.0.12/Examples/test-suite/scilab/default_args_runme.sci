exec("swigtest.start", -1);

checkequal(anonymous(), 7771, "anonymous()");
checkequal(anonymous(1234), 1234, "anonymous(1234)");

checkequal(booltest(), %T, "booltest()");
checkequal(booltest(%T), %T, "booltest(%T)");
checkequal(booltest(%F), %F, "booltest(%T)");

ec = new_EnumClass();
checkequal(EnumClass_blah(ec), %T, "EnumClass_blah(ec)");

checkequal(casts1(), [], "casts1()");
checkequal(casts1("Ciao"), "Ciao", "casts1(""Ciao"")");
checkequal(casts2(), "Hello", "casts2()");
checkequal(chartest1(), 'x', "chartest1()");
checkequal(chartest2(), '', "chartest2()");
checkequal(chartest1('y'), 'y', "chartest1(''y'')");
checkequal(reftest1(), 42, "reftest1()");
checkequal(reftest1(400), 400, "reftest1(400)");
checkequal(reftest2(), "hello", "reftest2()");

// Rename
f = new_Foo();
Foo_newname(f);
Foo_newname(f, 10);
Foo_renamed3arg(f, 10, 10.0);
Foo_renamed2arg(f, 10);
Foo_renamed1arg(f);
delete_Foo(f);

// Static functions
checkequal(Statics_staticmethod(), 10+20+30, "Statics_staticmethod()");
checkequal(Statics_staticmethod(100), 100+20+30, "Statics_staticmethod(100)");
checkequal(Statics_staticmethod(100, 200, 300), 100+200+300, "Statics_staticmethod(100, 200, 300)");

tricky = new_Tricky();
checkequal(Tricky_privatedefault(tricky), 200, "Tricky_privatedefault(tricky)");
checkequal(Tricky_protectedint(tricky), 2000, "Tricky_protectedint(tricky)");
checkequal(Tricky_protecteddouble(tricky), 987.654, "Tricky_protecteddouble(tricky)");
checkequal(Tricky_functiondefault(tricky), 500, "Tricky_functiondefault(tricky)");
checkequal(Tricky_contrived(tricky), 'X', "Tricky_contrived(tricky)");
delete_Tricky(tricky);

// Default argument is a constructor
k = constructorcall();
checkequal(Klass_val_get(k), -1, "Klass_constructorcall()");
delete_Klass(k);
k = constructorcall(new_Klass(2222));
checkequal(Klass_val_get(k), 2222, "Klass_constructorcall(new Klass(2222)");
delete_Klass(k);
k = constructorcall(new_Klass());
checkequal(Klass_val_get(k), -1, "Klass_constructorcall(new_Klass()");
delete_Klass(k);

// Const methods
cm = new_ConstMethods();
checkequal(ConstMethods_coo(cm), 20, "ConstMethods_coo()");
checkequal(ConstMethods_coo(cm, 1.0), 20, "ConstMethods_coo(1.0)");

// C linkage (extern "C")
checkequal(cfunc1(1), 2, "cfunc1(1)");
checkequal(cfunc2(1), 3, "cfunc2(1)");
checkequal(cfunc3(1), 4, "cfunc3(1)");

exec("swigtest.quit", -1);

