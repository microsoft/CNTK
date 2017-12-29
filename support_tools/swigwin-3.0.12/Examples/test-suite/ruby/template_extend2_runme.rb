#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'template_extend2'

a = Template_extend2::LBaz.new
b = Template_extend2::DBaz.new

raise RuntimeError unless a.foo() == "lBaz::foo"
raise RuntimeError unless b.foo() == "dBaz::foo"

