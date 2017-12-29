inherit_missing

a = inherit_missing.new_Foo();
b = inherit_missing.Bar();
c = inherit_missing.Spam();

x = inherit_missing.do_blah(a);
if (!strcmp(x, "Foo::blah"))
    error("Whoa! Bad return %s", x)
endif

x = inherit_missing.do_blah(b);
if (!strcmp(x, "Bar::blah"))
    error("Whoa! Bad return %s", x)
endif

x = inherit_missing.do_blah(c);
if (!strcmp(x, "Spam::blah"))
    error("Whoa! Bad return %s", x)
endif

inherit_missing.delete_Foo(a);
