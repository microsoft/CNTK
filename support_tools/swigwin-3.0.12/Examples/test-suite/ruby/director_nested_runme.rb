#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'director_nested'

NoProtectedError = Kernel.const_defined?("NoMethodError") ? NoMethodError : NameError

class A < Director_nested::FooBar_int
  protected
    def do_step
      "A::do_step;"
    end
  
    def get_value
      "A::get_value"
    end
end

a = A.new

begin 
  a.do_advance
rescue NoProtectedError
end

raise RuntimeError if a.step != "Bar::step;Foo::advance;Bar::do_advance;A::do_step;"


class B < Director_nested::FooBar_int
  protected
    def do_advance
      "B::do_advance;" + do_step
    end

    def do_step
      "B::do_step;"
    end
  
    def get_value
      "B::get_value"
    end
end


b = B.new
raise RuntimeError if b.step != "Bar::step;Foo::advance;B::do_advance;B::do_step;"
