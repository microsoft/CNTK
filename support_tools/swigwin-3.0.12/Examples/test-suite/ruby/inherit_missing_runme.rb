#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'inherit_missing'

a = Inherit_missing.new_Foo()
b = Inherit_missing::Bar.new
c = Inherit_missing::Spam.new

x = Inherit_missing.do_blah(a)
if x != "Foo::blah"
  puts "Whoa! Bad return #{x}"
end

x = Inherit_missing.do_blah(b)
if x != "Bar::blah"
  puts "Whoa! Bad return #{x}"
end

x = Inherit_missing.do_blah(c)
if x != "Spam::blah"
  puts "Whoa! Bad return #{x}"
end
