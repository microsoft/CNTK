#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'abstract_inherit'

include Abstract_inherit

#
# Shouldn't be able to instantiate any of these classes
# since none of them implements the pure virtual function
# declared in the base class (Foo).
#

exceptionRaised = false
begin
  Foo.new
rescue NameError
  exceptionRaised = true
rescue TypeError
  # In Ruby 1.8 the exception raised is:
  # TypeError: allocator undefined for Abstract_inherit::Foo
	exceptionRaised = true
ensure
  swig_assert( "exceptionRaised", binding )
end

exceptionRaised = false
begin
  Bar.new
rescue NameError
  exceptionRaised = true
rescue TypeError
  # In Ruby 1.8 the exception raised is:
  # TypeError: allocator undefined for Abstract_inherit::Bar
	exceptionRaised = true
ensure
  swig_assert( "exceptionRaised", binding )
end

exceptionRaised = false
begin
  Spam.new
rescue NameError
  exceptionRaised = true
rescue TypeError
  # In Ruby 1.8 the exception raised is:
  # TypeError: allocator undefined for Abstract_inherit::Spam
	exceptionRaised = true
ensure
  swig_assert( "exceptionRaised", binding )
end


