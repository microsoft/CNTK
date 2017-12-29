#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'director_constructor'

include Director_constructor

class Test < Foo
  def initialize(i)
    super(i)
  end

  def doubleit()
    self.a = (self.a * 2)
  end

  def test
    3
  end
end

a = Test.new(5) #dies here

raise RuntimeError if a.getit != 5
raise RuntimeError if a.do_test != 3

a.doubleit
raise RuntimeError if a.getit != 10

