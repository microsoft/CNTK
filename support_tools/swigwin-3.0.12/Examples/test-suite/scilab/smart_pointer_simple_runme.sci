exec("swigtest.start", -1);

f = new_Foo();
b = new_Bar(f);

Bar_x_set(b, 3);
if Bar_x_get(b) <> 3 then swigtesterror(); end

fp = Bar___deref__(b);
Bar_x_set(b, 4);
if Bar_x_get(b) <> 4 then swigtesterror(); end

exec("swigtest.quit", -1);
