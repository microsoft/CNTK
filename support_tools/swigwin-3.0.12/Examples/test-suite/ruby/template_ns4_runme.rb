#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'template_ns4'

d = Template_ns4.make_Class_DD()
raise RuntimeError if d.test() != "test"
