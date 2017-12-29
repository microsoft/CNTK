#!/usr/bin/env ruby
#
# Put script description here.
#
# 
# 
# 
#

require 'swig_assert'
require 'argout'

include Argout

swig_assert_each_line(<<'EOF', binding)

t = new_intp
intp_assign(t, 5)
v = incp(t)
val = intp_value(t)
val == 6

t = new_intp
intp_assign(t, 5)
v = incr(t)
v == 5
val = intp_value(t)
val == 6

t = new_intp
intp_assign(t, 5)
v = inctr(t)
v == 5
val = intp_value(t)
val == 6

EOF

#
# @todo: how to use voidhandle and handle?
#

