from director_smartptr import *


class director_smartptr_MyBarFoo(Foo):

  def ping(self):
    return "director_smartptr_MyBarFoo.ping()"

  def pong(self):
    return "director_smartptr_MyBarFoo.pong();" + self.ping()

  def upcall(self, fooBarPtr):
    return "override;" + fooBarPtr.FooBarDo()

  def makeFoo(self):
    return Foo()

def check(got, expected):
  if (got != expected):
    raise RuntimeError, "Failed, got: " + got + " expected: " + expected

fooBar = FooBar()

myBarFoo = director_smartptr_MyBarFoo()
check(myBarFoo.ping(), "director_smartptr_MyBarFoo.ping()")
check(Foo.callPong(myBarFoo), "director_smartptr_MyBarFoo.pong();director_smartptr_MyBarFoo.ping()")
check(Foo.callUpcall(myBarFoo, fooBar), "override;Bar::Foo2::Foo2Bar()")

myFoo = myBarFoo.makeFoo()
check(myFoo.pong(), "Foo::pong();Foo::ping()")
check(Foo.callPong(myFoo), "Foo::pong();Foo::ping()")
check(myFoo.upcall(FooBar()), "Bar::Foo2::Foo2Bar()")

myFoo2 = Foo().makeFoo()
check(myFoo2.pong(), "Foo::pong();Foo::ping()")
check(Foo.callPong(myFoo2), "Foo::pong();Foo::ping()")
check(myFoo2.upcall(FooBar()), "Bar::Foo2::Foo2Bar()")
