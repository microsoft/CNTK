#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'array_member'

include Array_member

f = Foo.new
f.data = Array_member.global_data

0.upto(7) { |i|
  unless get_value(f.data, i) == get_value(Array_member.global_data, i)
    raise RuntimeError, "Bad array assignment"
  end
}

0.upto(7) { |i|
  set_value(f.data, i, -i)
}

Array_member.global_data = f.data

0.upto(7) { |i|
  unless get_value(f.data, i) == get_value(Array_member.global_data, i)
    raise RuntimeError, "Bad array assignment"
  end
}

