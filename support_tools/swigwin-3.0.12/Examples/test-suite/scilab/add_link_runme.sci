exec("swigtest.start", -1);

try
    foo = new_Foo();
catch
    swigtesterror();
end

try
    foo2 = Foo_blah(foo);
catch
    swigtesterror();
end

try
    delete_Foo(foo);
catch
    swigtesterror();
end

try
    delete_Foo(foo2);
catch
    swigtesterror();
end

exec("swigtest.quit", -1);
