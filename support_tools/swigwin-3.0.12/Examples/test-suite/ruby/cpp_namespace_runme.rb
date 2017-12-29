#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

# Note: This example assumes that namespaces are flattened
require 'cpp_namespace'

n = Cpp_namespace.fact(4)
if n != 24
  raise "Bad return value!"
end

if Cpp_namespace.Foo != 42
  raise "Bad variable value!"
end

t = Cpp_namespace::Test.new
if t.method() != "Test::method"
  raise "Bad method return value!"
end

if Cpp_namespace.do_method(t) != "Test::method"
  raise "Bad return value!"
end

if Cpp_namespace.do_method2(t) != "Test::method"
  raise "Bad return value!"
end
    
Cpp_namespace.weird("hello", 4)

t2 = Cpp_namespace::Test2.new
t3 = Cpp_namespace::Test3.new
t4 = Cpp_namespace::Test4.new
t5 = Cpp_namespace::Test5.new

if Cpp_namespace.foo3(42) != 42
  raise "Bad return value!"
end

if Cpp_namespace.do_method3(t2, 40) != "Test2::method"
  raise "Bad return value!"
end

if Cpp_namespace.do_method3(t3, 40) != "Test3::method"
  raise "Bad return value!"
end

if Cpp_namespace.do_method3(t4, 40) != "Test4::method"
  raise "Bad return value!"
end

if Cpp_namespace.do_method3(t5, 40) != "Test5::method"
  raise "Bad return value!"
end
