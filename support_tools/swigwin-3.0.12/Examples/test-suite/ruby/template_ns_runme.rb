#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'template_ns'

include Template_ns

p1 = Pairii.new(2, 3)
p2 = Pairii.new(p1)

raise RuntimeError if p2.first != 2
raise RuntimeError if p2.second != 3

p3 = Pairdd.new(3.5, 2.5)
p4 = Pairdd.new(p3)

raise RuntimeError if p4.first != 3.5
raise RuntimeError if p4.second != 2.5
