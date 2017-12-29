#!/usr/bin/env ruby
#
# Simple test of std::ostringstream.
#
# 
# 
# 
#

require 'swig_assert'

require 'li_std_stream'
include Li_std_stream

swig_assert_each_line(<<'EOF', binding)

a = A.new
o = Ostringstream.new
o << a << " " << 2345 << " " << 1.435
o.str == "A class 2345 1.435"

EOF
