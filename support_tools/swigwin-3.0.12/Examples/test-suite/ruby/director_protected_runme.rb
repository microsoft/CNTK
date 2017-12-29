#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'director_protected'

NoProtectedError = Kernel.const_defined?("NoMethodError") ? NoMethodError : NameError

class FooBar < Director_protected::Bar
  protected
    def ping
      "FooBar::ping();"
    end
end

class Hello < FooBar
  public
    def pang
      ping
    end
end

b = Director_protected::Bar.new
fb = FooBar.new

p = 0
begin 
  b.ping
  p = 1
rescue NoProtectedError
end

h = Hello.new

raise RuntimeError if p == 1
raise RuntimeError if b.pong != "Bar::pong();Foo::pong();Bar::ping();"
raise RuntimeError if fb.pong != "Bar::pong();Foo::pong();FooBar::ping();"
raise RuntimeError if h.pang != "FooBar::ping();"
