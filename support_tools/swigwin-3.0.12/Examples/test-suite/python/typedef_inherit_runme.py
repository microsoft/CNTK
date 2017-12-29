import typedef_inherit

a = typedef_inherit.Foo()
b = typedef_inherit.Bar()

x = typedef_inherit.do_blah(a)
if x != "Foo::blah":
    print "Whoa! Bad return", x

x = typedef_inherit.do_blah(b)
if x != "Bar::blah":
    print "Whoa! Bad return", x

c = typedef_inherit.Spam()
d = typedef_inherit.Grok()

x = typedef_inherit.do_blah2(c)
if x != "Spam::blah":
    print "Whoa! Bad return", x

x = typedef_inherit.do_blah2(d)
if x != "Grok::blah":
    print "Whoa! Bad return", x
