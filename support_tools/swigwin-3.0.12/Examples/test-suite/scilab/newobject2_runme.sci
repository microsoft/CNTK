exec("swigtest.start", -1);

try
    x = makeFoo();
catch
    swigtesterror();
end
if fooCount() <> 1 then swigtesterror(); end

try
    y = makeFoo();
catch
    swigtesterror();
end
if fooCount() <> 2 then swigtesterror(); end

try
    delete_Foo(x);
catch
    swigtesterror();
end
if fooCount() <> 1 then swigtesterror(); end

try
    delete_Foo(y);
catch
    swigtesterror();
end
if fooCount() <> 0 then swigtesterror(); end

exec("swigtest.quit", -1);
