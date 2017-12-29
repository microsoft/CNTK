#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'rename_scope'

include Rename_scope

a = Natural_UP.new
b = Natural_BP.new

raise RuntimeError if a.rtest() != 1

raise RuntimeError if b.rtest() != 1
