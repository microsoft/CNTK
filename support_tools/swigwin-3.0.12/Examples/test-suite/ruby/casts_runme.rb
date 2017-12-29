#!/usr/bin/env ruby
#
# Put script description here.
#
# 
# 
# 
#

require 'swig_assert'
require 'casts'

include Casts

swig_assert( B.ancestors.include?(A), 'B.ancestors.include? A' )

a = A.new
a.hello

b = B.new
b.hello

swig_assert( b.kind_of?( A ), ' B.kind_of? A' )
