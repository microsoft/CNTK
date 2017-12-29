#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'director_unroll'

class MyFoo < Director_unroll::Foo
  def ping
    "MyFoo::ping()"
  end
end

a = MyFoo.new

b = Director_unroll::Bar.new

b.set(a)
c = b.get()

raise RuntimeError if a != c

