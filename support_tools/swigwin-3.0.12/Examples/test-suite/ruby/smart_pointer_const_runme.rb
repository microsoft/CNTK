#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'smart_pointer_const'

include Smart_pointer_const

f = Foo.new
b = Bar.new(f)

b.x = 3
raise RuntimeError if b.getx() != 3

fp = b.__deref__()
fp.x = 4
raise RuntimeError if fp.getx() != 4
