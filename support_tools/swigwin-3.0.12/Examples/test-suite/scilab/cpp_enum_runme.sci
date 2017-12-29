exec("swigtest.start", -1);

f = new_Foo();

if Foo_hola_get(f) <> Hello_get() then swigtesterror("Foo_hola_get() <> ""Hello"""); end

Foo_hola_set(f, Hi_get());
if Foo_hola_get(f) <> Hi_get() then swigtesterror("Foo_hola_get() <> ""Hi"""); end

exec("swigtest.quit", -1);
