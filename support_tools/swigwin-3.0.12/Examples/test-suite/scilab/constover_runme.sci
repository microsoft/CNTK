exec("swigtest.start", -1);

p = test("test");
if strcmp(p, "test") <> 0 then swigtesterror(); end

p = test_pconst("test");
if strcmp(p, "test_pconst") <> 0 then swigtesterror(); end

f = new_Foo();
p = Foo_test(f, "test");
if strcmp(p,"test") <> 0 then swigtesterror(); end

p = Foo_test_pconst(f, "test");
if strcmp(p,"test_pconst") <> 0 then swigtesterror(); end

p = Foo_test_constm(f, "test");
if strcmp(p,"test_constmethod") <> 0 then swigtesterror(); end

p = Foo_test_pconstm(f, "test");
if strcmp(p,"test_pconstmethod") <> 0 then swigtesterror(); end

exec("swigtest.quit", -1);
