#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'li_std_pair_lang_object'
include Li_std_pair_lang_object

swig_assert_each_line(<<'EOF', binding)
val = ValuePair.new
val.first = 'sd'
val.second = [5,4,3]
EOF
