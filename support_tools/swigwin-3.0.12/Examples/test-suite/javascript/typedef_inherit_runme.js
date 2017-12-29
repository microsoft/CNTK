var typedef_inherit = require("typedef_inherit");

a = new typedef_inherit.Foo();
b = new typedef_inherit.Bar();

x = typedef_inherit.do_blah(a);
if (x != "Foo::blah")
    print("Whoa! Bad return" + x);

x = typedef_inherit.do_blah(b);
if (x != "Bar::blah")
    print("Whoa! Bad return" + x);

c = new typedef_inherit.Spam();
d = new typedef_inherit.Grok();

x = typedef_inherit.do_blah2(c);
if (x != "Spam::blah")
    print("Whoa! Bad return" + x);

x = typedef_inherit.do_blah2(d);
if (x != "Grok::blah")
    print ("Whoa! Bad return" + x);
