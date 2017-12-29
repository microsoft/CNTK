#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'director_abstract'

class MyFoo < Director_abstract::Foo
  def ping
    "MyFoo::ping()"
  end
end


a = MyFoo.new

if a.ping != "MyFoo::ping()"
  raise RuntimeError, a.ping
end

if a.pong != "Foo::pong();MyFoo::ping()"
  raise RuntimeError, a.pong
end


class MyExample1 < Director_abstract::Example1
  def color(r,g,b)
    r
  end
end

#m1 = MyExample1.new
#
#if m1.color(1,2,3) != 1
#  raise RuntimeError, m1.color
#end
