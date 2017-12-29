exec("swigtest.start", -1);

try
    foo = new_Foo();
catch
    swigtesterror();
end

checkequal(Foo_x_get(foo), 0, "Foo_x_get()");

checkequal(Foo_y_get(foo), 0, "Foo_y_get()");

checkequal(Foo_z_get(foo), 0, "Foo_y_get()");

checkequal(Foo_f_get(foo), 0, "Foo_f_get()");

checkequal(Foo_seq_get(foo), 0, "Foo_seq_get()");

try
    Foo_x_set(foo, 5);
catch
    swigtesterror();
end
checkequal(Foo_x_get(foo), 5, "Foo_x_get()");

try
    Foo_y_set(foo, 5);
catch
    swigtesterror();
end
checkequal(Foo_y_get(foo), 5, "Foo_y_get()");

try
    Foo_f_set(foo, 1);
catch
    swigtesterror();
end
checkequal(Foo_f_get(foo), 1, "Foo_f_get()");

try
    Foo_z_set(foo, 13);
catch
    swigtesterror();
end
checkequal(Foo_z_get(foo), 13, "Foo_z_get()");

try
    Foo_seq_set(foo, 3);
catch
    swigtesterror();
end
checkequal(Foo_seq_get(foo), 3, "Foo_seq_get()");

try
    delete_Foo(foo);
catch
    swigtesterror();
end

exec("swigtest.quit", -1);
