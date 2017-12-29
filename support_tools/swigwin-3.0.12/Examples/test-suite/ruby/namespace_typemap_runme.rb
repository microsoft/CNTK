#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'namespace_typemap'

include Namespace_typemap

raise RuntimeError if stest1("hello") != "hello"

raise RuntimeError if stest2("hello") != "hello"

raise RuntimeError if stest3("hello") != "hello"

raise RuntimeError if stest4("hello") != "hello"

raise RuntimeError if stest5("hello") != "hello"

raise RuntimeError if stest6("hello") != "hello"

raise RuntimeError if stest7("hello") != "hello"

raise RuntimeError if stest8("hello") != "hello"

raise RuntimeError if stest9("hello") != "hello"

raise RuntimeError if stest10("hello") != "hello"

raise RuntimeError if stest11("hello") != "hello"

raise RuntimeError if stest12("hello") != "hello"

begin
  ttest1(-14)
  raise RuntimeError
rescue RangeError
end
