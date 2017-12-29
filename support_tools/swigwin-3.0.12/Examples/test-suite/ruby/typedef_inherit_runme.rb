#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'typedef_inherit'

a = Typedef_inherit::Foo.new
b = Typedef_inherit::Bar.new

x = Typedef_inherit.do_blah(a)
if x != "Foo::blah"
  puts "Whoa! Bad return #{x}"
end

x = Typedef_inherit.do_blah(b)
if x != "Bar::blah"
  puts "Whoa! Bad return #{x}"
end

c = Typedef_inherit::Spam.new
d = Typedef_inherit::Grok.new

x = Typedef_inherit.do_blah2(c)
if x != "Spam::blah"
  puts "Whoa! Bad return #{x}"
end

x = Typedef_inherit.do_blah2(d)
if x != "Grok::blah"
  puts "Whoa! Bad return #{x}"
end
