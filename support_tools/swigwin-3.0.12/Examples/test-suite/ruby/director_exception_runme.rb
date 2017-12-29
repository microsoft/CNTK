#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'director_exception'

include Director_exception

class MyFoo < Foo
  def ping
    raise NotImplementedError, "MyFoo::ping() EXCEPTION"
  end
end

class MyFoo2 < Foo
  def ping
    nil # error: should return a string
  end
end

class MyFoo3 < Foo
  def ping
    5 # error: should return a string
  end
end

ok = false

a = MyFoo.new
b = launder(a)

begin
  b.pong
rescue NotImplementedError
  ok = true
end

raise RuntimeError unless ok

ok = false

a = MyFoo2.new
b = launder(a)

begin
  b.pong
rescue TypeError
  ok = true
end


a = MyFoo3.new
b = launder(a)

begin
  b.pong
rescue TypeError
  ok = true
end


raise RuntimeError unless ok

