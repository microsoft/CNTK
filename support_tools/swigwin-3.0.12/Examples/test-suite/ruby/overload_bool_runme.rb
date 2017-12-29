#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'overload_bool'

include Overload_bool

# Overloading bool, int, string
if overloaded(true) != "bool"
  raise RuntimeError, "wrong!"
end
if overloaded(false) != "bool"
  raise RuntimeError, "wrong!"
end

if overloaded(0) != "int"
  raise RuntimeError, "wrong!"
end
if overloaded(1) != "int"
  raise RuntimeError, "wrong!"
end
if overloaded(2) != "int"
  raise RuntimeError, "wrong!"
end

if overloaded("1234") != "string"
  raise RuntimeError, "wrong!"
end

# Test bool masquerading as integer
# Not possible

# Test int masquerading as bool
if boolfunction(0) != "false"
  raise RuntimeError, "wrong!"
end
if boolfunction(1) != "true"
  raise RuntimeError, "wrong!"
end
if boolfunction(2) != "true"
  raise RuntimeError, "wrong!"
end

#############################################

# Overloading bool, int, string
if overloaded_ref(true) != "bool"
  raise RuntimeError, "wrong!"
end
if overloaded_ref(false) != "bool"
  raise RuntimeError, "wrong!"
end

if overloaded_ref(0) != "int"
  raise RuntimeError, "wrong!"
end
if overloaded_ref(1) != "int"
  raise RuntimeError, "wrong!"
end
if overloaded_ref(2) != "int"
  raise RuntimeError, "wrong!"
end

if overloaded_ref("1234") != "string"
  raise RuntimeError, "wrong!"
end

# Test bool masquerading as integer
# Not possible

# Test int masquerading as bool
if boolfunction_ref(0) != "false"
  raise RuntimeError, "wrong!"
end
if boolfunction_ref(1) != "true"
  raise RuntimeError, "wrong!"
end
if boolfunction_ref(2) != "true"
  raise RuntimeError, "wrong!"
end
