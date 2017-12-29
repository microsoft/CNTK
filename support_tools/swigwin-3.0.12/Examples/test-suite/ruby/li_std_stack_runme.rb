#!/usr/bin/env ruby
#
# A simple std::stack test
#
# 
# 
# 
#

require 'swig_assert'

require 'li_std_stack'
include Li_std_stack

swig_assert_each_line(<<'EOF', binding)
a = IntStack.new
a << 1
a << 2
a << 3
a.top == 3
a.pop
a.top == 2
a.pop
a.top == 1
a.pop
a.size == 0
a.empty? == true
# a.top == Qnil

EOF
