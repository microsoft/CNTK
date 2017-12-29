#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'director_smartptr'

include Director_smartptr

class Director_smartptr_MyBarFoo < Foo

  def ping()
    return "director_smartptr_MyBarFoo.ping()"
  end

  def pong()
    return "director_smartptr_MyBarFoo.pong();" + ping()
  end

  def upcall(fooBarPtr)
    return "override;" + fooBarPtr.FooBarDo()
  end

  def makeFoo()
    return Foo.new()
  end
end

def check(got, expected)
  if (got != expected)
    raise RuntimeError, "Failed, got: #{got} expected: #{expected}"
  end
end

fooBar = Director_smartptr::FooBar.new()

myBarFoo = Director_smartptr_MyBarFoo.new()
check(myBarFoo.ping(), "director_smartptr_MyBarFoo.ping()")
check(Foo.callPong(myBarFoo), "director_smartptr_MyBarFoo.pong();director_smartptr_MyBarFoo.ping()")
check(Foo.callUpcall(myBarFoo, fooBar), "override;Bar::Foo2::Foo2Bar()")

myFoo = myBarFoo.makeFoo()
check(myFoo.pong(), "Foo::pong();Foo::ping()")
check(Foo.callPong(myFoo), "Foo::pong();Foo::ping()")
check(myFoo.upcall(FooBar.new()), "Bar::Foo2::Foo2Bar()")

myFoo2 = Foo.new().makeFoo()
check(myFoo2.pong(), "Foo::pong();Foo::ping()")
check(Foo.callPong(myFoo2), "Foo::pong();Foo::ping()")
check(myFoo2.upcall(FooBar.new()), "Bar::Foo2::Foo2Bar()")
