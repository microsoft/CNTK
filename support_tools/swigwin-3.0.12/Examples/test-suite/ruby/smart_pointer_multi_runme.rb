#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'smart_pointer_multi'

include Smart_pointer_multi

f = Foo.new
b = Bar.new(f)
s = Spam.new(b)
g = Grok.new(b)

s.x = 3
raise RuntimeError if s.getx() != 3

g.x = 4
raise RuntimeError if g.getx() != 4

