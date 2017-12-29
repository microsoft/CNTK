#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'overload_copy'

include Overload_copy

f = Foo.new
g = Foo.new(f)
