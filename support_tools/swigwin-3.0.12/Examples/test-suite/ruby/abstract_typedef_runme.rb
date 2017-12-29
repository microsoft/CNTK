#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'abstract_typedef'

include Abstract_typedef

swig_assert_each_line(<<'EOF', binding)

e = Engine.new
a = A.new
a.write(e)

EOF

