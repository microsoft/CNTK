#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'
require 'abstract_signature'

include Abstract_signature

#
# Shouldn't be able to instantiate Abstract_foo, because it declares
# a pure virtual function.
#

exceptionRaised = false
begin
  foo = Abstract_foo.new
  begin
    foo.meth(1)
  rescue RuntimeError 
    # here we are using directors
    exceptionRaised = true
  end
rescue NameError
  exceptionRaised = true
rescue TypeError
  # In Ruby 1.8 the exception raised is:
  # TypeError: allocator undefined for Abstract_signature::Abstract_foo
	exceptionRaised = true
ensure
  swig_assert( "exceptionRaised", binding)
end

#
# Shouldn't be able to instantiate an Abstract_bar either, because it doesn't
# implement the pure virtual function with the correct signature.
#

exceptionRaised = false
begin
  bar = Abstract_bar.new
  begin
    bar.meth(1)
  rescue RuntimeError 
    # here we are using directors
    exceptionRaised = true
  end
rescue NameError
  exceptionRaised = true
rescue TypeError
  # In Ruby 1.8 the exception raised is:
  # TypeError: allocator undefined for Abstract_signature::Abstract_bar
	exceptionRaised = true
ensure
  swig_assert( "exceptionRaised", binding)
end

