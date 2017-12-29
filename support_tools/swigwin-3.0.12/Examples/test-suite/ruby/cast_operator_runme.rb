#!/usr/bin/env ruby
#
# Put script description here.
#
# 
# 
# 
#

require 'swig_assert'
require 'cast_operator'
include Cast_operator

a = A.new
t = a.tochar

swig_assert( t == 'hi' )
