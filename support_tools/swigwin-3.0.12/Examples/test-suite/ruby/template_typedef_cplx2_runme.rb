#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'template_typedef_cplx2'

include Template_typedef_cplx2

#
# double case
#


d = make_Identity_double()
raise RuntimeError, "#{d} is not an instance" if d.is_a? SWIG::Pointer

classname = d.class.name.split(/::/).last
unless classname =~ /ArithUnaryFunction/
  raise RuntimeError, "#{d} is not an ArithUnaryFunction"
end

e = make_Multiplies_double_double_double_double(d, d)
raise RuntimeError, "#{e} is not an instance" if e.is_a? SWIG::Pointer

classname = e.class.name.split(/::/).last
unless classname =~ /ArithUnaryFunction/
  raise RuntimeError, "#{e} is not an ArithUnaryFunction"
end


#
# complex case
#

c = make_Identity_complex()
raise RuntimeError, "#{c}is not an instance" if c.is_a? SWIG::Pointer

classname = c.class.name.split(/::/).last
unless classname =~ /ArithUnaryFunction/
  raise RuntimeError, "#{c} is not an ArithUnaryFunction"
end

f = make_Multiplies_complex_complex_complex_complex(c, c)
raise RuntimeError, "{f} is not an instance" if f.is_a? SWIG::Pointer

classname = f.class.name.split(/::/).last
unless classname =~ /ArithUnaryFunction/
  raise RuntimeError, "#{f} is not an ArithUnaryFunction"
end

#
# Mix case
#

g = make_Multiplies_double_double_complex_complex(d, c)
raise RuntimeError, "#{g} is not an instance" if g.is_a? SWIG::Pointer

classname = g.class.name.split(/::/).last
unless classname =~ /ArithUnaryFunction/
  raise RuntimeError, "#{g} is not an ArithUnaryFunction"
end

# This should raise NoMethodError if it fails
g.get_value()
  
h = make_Multiplies_complex_complex_double_double(c, d)
raise RuntimeError, "#{h} is not an instance" if h.is_a? SWIG::Pointer

classname = h.class.name.split(/::/).last
unless classname =~ /ArithUnaryFunction/
  raise RuntimeError, "#{h} is not an ArithUnaryFunction"
end


