#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'li_std_string'

include Li_std_string

# Checking expected use of %typemap(in) std::string {}
test_value("Fee")

# Checking expected result of %typemap(out) std::string {}
raise RuntimeError unless test_value("Fi") == "Fi"

# Verify type-checking for %typemap(in) std::string {}
exceptionRaised = false
begin
  test_value(0)
rescue TypeError
  exceptionRaised = true
ensure
  raise RuntimeError unless exceptionRaised
end

# Checking expected use of %typemap(in) const std::string & {}
test_const_reference("Fo")

# Checking expected result of %typemap(out) const std::string& {}
raise RuntimeError unless test_const_reference("Fum") == "Fum"

# Verify type-checking for %typemap(in) const std::string & {}
exceptionRaised = false
begin
  test_const_reference(0)
rescue TypeError
  exceptionRaised = true
ensure
  raise RuntimeError unless exceptionRaised
end

#
# Input and output typemaps for pointers and non-const references to
# std::string are *not* supported; the following tests confirm
# that none of these cases are slipping through.
#

exceptionRaised = false
begin
  test_pointer("foo")
rescue TypeError
  exceptionRaised = true
ensure
  raise RuntimeError unless exceptionRaised
end

result = test_pointer_out()
raise RuntimeError if result.is_a? String

exceptionRaised = false
begin
  test_const_pointer("bar")
rescue TypeError
  exceptionRaised = true
ensure
  raise RuntimeError unless exceptionRaised
end

result = test_const_pointer_out()
raise RuntimeError if result.is_a? String

exceptionRaised = false
begin
  test_reference("foo")
rescue TypeError
  exceptionRaised = true
ensure
  raise RuntimeError unless exceptionRaised
end

result = test_reference_out()
raise RuntimeError if result.is_a? String


# Member Strings
myStructure = Structure.new
if (myStructure.MemberString2 != "member string 2")
  raise RuntimeError
end
s = "Hello"
myStructure.MemberString2 = s
if (myStructure.MemberString2 != s)
  raise RuntimeError
end
if (myStructure.ConstMemberString != "const member string")
  raise RuntimeError 
end


if (Structure.StaticMemberString2 != "static member string 2")
  raise RuntimeError
end
Structure.StaticMemberString2 = s
if (Structure.StaticMemberString2 != s)
  raise RuntimeError
end
if (Structure.ConstStaticMemberString != "const static member string")
  raise RuntimeError
end


if (test_reference_input("hello") != "hello")
  raise RuntimeError
end
s = test_reference_inout("hello")
if (s != "hellohello")
  raise RuntimeError
end


if (stdstring_empty() != "")
  raise RuntimeError
end

if (c_empty() != "") 
  raise RuntimeError
end


if (c_null() != nil) 
  raise RuntimeError
end


if (get_null(c_null()) != nil) 
  raise RuntimeError
end

