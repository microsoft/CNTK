#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'varargs'

if Varargs.test("Hello") != "Hello"
  raise RuntimeError, "Failed"
end

f = Varargs::Foo.new("Greetings")
if f.str != "Greetings"
  raise RuntimeError, "Failed"
end

if f.test("Hello") != "Hello"
  raise RuntimeError, "Failed"
end

