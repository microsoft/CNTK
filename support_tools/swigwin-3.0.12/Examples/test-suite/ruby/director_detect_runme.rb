#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'director_detect'

class MyBar < Director_detect::Bar
  def initialize(v)
    @val = v
  end

  def get_value
    @val = @val + 1
  end
  
  def get_class
    @val = @val + 1
    Director_detect::A
  end

  def just_do_it
    @val = @val + 1
  end

  def clone
    MyBar.new(@val)
  end

  def val
    @val
  end
end


b = MyBar.new(2)

f = b

v = f.get_value
a = f.get_class
f.just_do_it

c = b.clone
vc = c.get_value

raise RuntimeError if (v != 3) || (b.val != 5) || (vc != 6)

