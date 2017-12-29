# Ported to C# li_attribute_runme.cs

import li_attribute

aa = li_attribute.A(1, 2, 3)

if aa.a != 1:
    raise RuntimeError
aa.a = 3
if aa.a != 3:
    print aa.a
    raise RuntimeError

if aa.b != 2:
    print aa.b
    raise RuntimeError
aa.b = 5
if aa.b != 5:
    raise RuntimeError

if aa.d != aa.b:
    raise RuntimeError

if aa.c != 3:
    raise RuntimeError
#aa.c = 5
# if aa.c != 3:
#  raise RuntimeError

pi = li_attribute.Param_i(7)
if pi.value != 7:
    raise RuntimeError

pi.value = 3
if pi.value != 3:
    raise RuntimeError

b = li_attribute.B(aa)

if b.a.c != 3:
    raise RuntimeError

# class/struct attribute with get/set methods using return/pass by reference
myFoo = li_attribute.MyFoo()
myFoo.x = 8
myClass = li_attribute.MyClass()
myClass.Foo = myFoo
if myClass.Foo.x != 8:
    raise RuntimeError

# class/struct attribute with get/set methods using return/pass by value
myClassVal = li_attribute.MyClassVal()
if myClassVal.ReadWriteFoo.x != -1:
    raise RuntimeError
if myClassVal.ReadOnlyFoo.x != -1:
    raise RuntimeError
myClassVal.ReadWriteFoo = myFoo
if myClassVal.ReadWriteFoo.x != 8:
    raise RuntimeError
if myClassVal.ReadOnlyFoo.x != 8:
    raise RuntimeError

# string attribute with get/set methods using return/pass by value
myStringyClass = li_attribute.MyStringyClass("initial string")
if myStringyClass.ReadWriteString != "initial string":
    raise RuntimeError
if myStringyClass.ReadOnlyString != "initial string":
    raise RuntimeError
myStringyClass.ReadWriteString = "changed string"
if myStringyClass.ReadWriteString != "changed string":
    raise RuntimeError
if myStringyClass.ReadOnlyString != "changed string":
    raise RuntimeError

# Check a proper AttributeError is raised for non-existent attributes, old versions used to raise unhelpful error:
# AttributeError: type object 'object' has no attribute '__getattr__'
try:
    x = myFoo.does_not_exist
    raise RuntimeError
except AttributeError, e:
    if str(e).find("does_not_exist") == -1:
        raise RuntimeError

