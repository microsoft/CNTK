#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'smart_pointer_rename'

include Smart_pointer_rename

f = Foo.new
b = Bar.new(f)

raise RuntimeError if b.test() != 3

raise RuntimeError if b.ftest1(1) != 1

raise RuntimeError if b.ftest2(2,3) != 2
