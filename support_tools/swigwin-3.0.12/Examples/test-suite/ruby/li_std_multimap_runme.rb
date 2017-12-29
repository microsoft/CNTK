#!/usr/bin/env ruby
#
# Tests for std::multimap
#
# 
# 
# 
#

require 'swig_assert'
require 'li_std_multimap'

swig_assert_each_line(<<'EOF', binding)

a1 = Li_std_multimap::A.new(3)
a2 = Li_std_multimap::A.new(7)

m = Li_std_multimap::MultimapA.new
m[0] = a1
m[0] = a2
m[0].size == 2
m.respond_to?(:each) == true
m.respond_to?(:each_key) == true
m.respond_to?(:each_value) == true
m.values_at(0)[0] == m[0]
EOF

