#!/usr/bin/env ruby
#
# Runtime tests for enums.i
#

require 'swig_assert'
require 'enums'

swig_assert_each_line( <<EOF )
Enums::CSP_ITERATION_FWD == 0
Enums::CSP_ITERATION_BWD == 11
Enums::ABCDE == 0
Enums::FGHJI == 1
Enums.bar1(1)
Enums.bar2(1)
Enums.bar3(1)
Enums::Boo == 0
Enums::Hoo == 5
Enums::Globalinstance1 == 0
Enums::Globalinstance2 == 1
Enums::Globalinstance3 == 30
Enums::AnonEnum1 == 0
Enums::AnonEnum2 == 100
Enums::BAR1 == 0
Enums::BAR2 == 1
EOF

#
# @bug: 
#
# swig_assert_each_line( <<EOF )
# Enums::IFoo::Phoo == 50
# Enums::IFoo::Char == 'a'[0]
# EOF
