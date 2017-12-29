#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'abstract_inherit_ok'

include Abstract_inherit_ok

#
# Shouldn't be able to instantiate Foo, because it declares
# a pure virtual function.
#

exceptionRaised = false
begin
  Foo.new
rescue NameError
  exceptionRaised = true
rescue TypeError
  # In Ruby 1.8 the exception raised is:
  # TypeError: allocator undefined for Abstract_inherit_ok::Foo
	exceptionRaised = true
ensure
  swig_assert( "exceptionRaised", binding )
end

#
# This one's OK since we cleared it with a %feature("notabstract")
# declaration in the interface file.
#

exceptionRaised = false
begin
  spam = Spam.new
  raise RuntimeError unless spam.blah == 0 
rescue NameError
  exceptionRaised = true
ensure
  swig_assert( "!exceptionRaised", binding )
end

