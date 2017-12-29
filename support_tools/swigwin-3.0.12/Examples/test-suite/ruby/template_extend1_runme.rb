#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'template_extend1'

a = Template_extend1::LBaz.new
b = Template_extend1::DBaz.new

raise RuntimeError unless a.foo() == "lBaz::foo"
raise RuntimeError unless b.foo() == "dBaz::foo"

