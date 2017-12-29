import dynamic_cast

f = dynamic_cast.Foo()
b = dynamic_cast.Bar()

x = f.blah()
y = b.blah()

a = dynamic_cast.do_test(y)
if a != "Bar::test":
    print "Failed!!"
