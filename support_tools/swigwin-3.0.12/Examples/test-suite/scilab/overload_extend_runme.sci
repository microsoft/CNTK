exec("swigtest.start", -1);

//try
//    x = new_Foo();
//catch
//    swigtesterror();
//end
//if Foo_test(x) <> 0 then swigtesterror(); end
//if Foo_test(x, 1) <> 1 then swigtesterror(); end
//if Foo_test(x, 2, 3) <> 5 then swigtesterror(); end
//if Foo_test(x, "Hello, swig!") <> 2 then swigtesterror(); end

exec("swigtest.quit", -1);
