exec("swigtest.start", -1);

foo = new_Foo();
Foo_x_set(foo, 1);
if Foo_x_get(foo) <> 1 then swigtesterror(); end

bar = new_Bar();
Bar_a_set(bar, foo);
a = Bar_a_get(bar);
if Foo_x_get(a) <> 1 then swigtesterror(); end

Bar_b_set(bar, foo);
b = Bar_b_get(bar);
if Foo_x_get(b) <> 1 then swigtesterror(); end

exec("swigtest.quit", -1);
