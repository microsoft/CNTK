#!/usr/bin/env ruby
#
#

require 'swig_assert'

require 'const_const'
include Const_const

swig_assert_each_line <<EOF
foo(1)  # 1 is unused
EOF


