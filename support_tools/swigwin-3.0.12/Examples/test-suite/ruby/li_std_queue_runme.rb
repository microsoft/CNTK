#!/usr/bin/env ruby
#
# A simple std::queue test
#
# 
# 
# 
#

require 'swig_assert'

require 'li_std_queue'
include Li_std_queue

swig_assert_each_line(<<'EOF', binding)
a = IntQueue.new
a << 1
a << 2
a << 3
a.back  == 3
a.front == 1
a.pop
a.back  == 3
a.front == 2
a.pop
a.back  == 3
a.front == 3
a.pop
a.size == 0
a.empty? == true

EOF
