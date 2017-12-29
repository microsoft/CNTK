#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'virtual_derivation'

b = Virtual_derivation::B.new 3

if b.get_a() != b.get_b() 
  print "something is still wrong ", b.get_a(), b.get_b(), "\n"
end
