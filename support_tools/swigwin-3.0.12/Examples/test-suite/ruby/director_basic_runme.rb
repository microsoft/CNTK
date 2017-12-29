#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'director_basic'

class MyFoo < Director_basic::Foo
  def ping
    "MyFoo::ping()"
  end
end

a = MyFoo.new

raise RuntimeError if a.ping != "MyFoo::ping()"
raise RuntimeError if a.pong != "Foo::pong();MyFoo::ping()"

b = Director_basic::Foo.new

raise RuntimeError if b.ping != "Foo::ping()"
raise RuntimeError if b.pong != "Foo::pong();Foo::ping()"


a = Director_basic::MyClass.new 
a = Director_basic::MyClassT_i.new 


a = Director_basic::MyClass.new 1
a = Director_basic::MyClassT_i.new 1
