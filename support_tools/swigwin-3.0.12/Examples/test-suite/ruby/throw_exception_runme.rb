#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'throw_exception'

include Throw_exception

foo = Foo.new

begin
  foo.test_int
rescue => ex
  raise RuntimeError if ex.message != "37"
end

begin
  foo.test_msg
rescue => ex
  raise RuntimeError if ex.message != "Dead"
end

begin
  foo.test_multi(1)
rescue => ex
  raise RuntimeError if ex.message != "37"
end

begin
  foo.test_multi(2)
rescue => ex
  raise RuntimeError if ex.message != "Dead"
end

