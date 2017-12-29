#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'ruby_naming'

# Check class names
if not Ruby_naming
  raise RuntimeError, 'Invalid module name for Ruby_naming'
end

if not Ruby_naming::MyClass
  raise RuntimeError, 'Invalid class name for MyClass'
end


# Check constant names / values
if Ruby_naming::CONSTANT1 != 1
  raise RuntimeError, "Incorrect value for CONSTANT1" 
end

if Ruby_naming::CONSTANT2 != 2
  raise RuntimeError, "Incorrect value for CONSTANT2" 
end

# Check constant names / values
if Ruby_naming::CONSTANT3 != 3
  raise RuntimeError, "Incorrect value for CONSTANT3" 
end

if not (Ruby_naming::methods.include?("constant4") || Ruby_naming::methods.include?(:constant4))
  raise RuntimeError, "Incorrect mapping for constant4" 
end

if not (Ruby_naming::methods.include?("constant5") || Ruby_naming::methods.include?(:constant5))
  raise RuntimeError, "Incorrect mapping for constant5" 
end

if not (Ruby_naming::methods.include?("constant6") || Ruby_naming::methods.include?(:constant6))
  raise RuntimeError, "Incorrect mapping for constant6" 
end

if not (Ruby_naming::TestConstants.instance_methods.include?("constant7") || Ruby_naming::TestConstants.instance_methods.include?(:constant7))
  raise RuntimeError, "Incorrect mapping for constant7" 
end

if not (Ruby_naming::TestConstants.methods.include?("constant8") || Ruby_naming::TestConstants.methods.include?(:constant8))
  raise RuntimeError, "Incorrect mapping for constant8" 
end

# There is no constant9 because it is illegal C++
#if not Ruby_naming::TestConstants.instance_methods.include?("constant9")
#  raise RuntimeError, "Incorrect mapping for constant9" 
#end

if Ruby_naming::TestConstants::CONSTANT10 != 10
  raise RuntimeError, "Incorrect value for CONSTANT10" 
end

if not (Ruby_naming::methods.include?("constant11") || Ruby_naming::methods.include?(:constant11))
  raise RuntimeError, "Incorrect mapping for constant11" 
end


# Check enums
if Ruby_naming::constants.include?("Color")
  raise RuntimeError, "Color enum should not be exposed to Ruby" 
end

if Ruby_naming::Red != 0
  raise RuntimeError, "Incorrect value for enum RED" 
end

if Ruby_naming::Green != 1
  raise RuntimeError, "Incorrect value for enum GREEN" 
end

if Ruby_naming::Blue != 2
  raise RuntimeError, "Incorrect value for enum BLUE" 
end


# Check method names
my_class = Ruby_naming::MyClass.new()

if my_class.method_one != 1 
  raise RuntimeError, "Incorrect value for method_one" 
end
  
if my_class.method_two != 2
  raise RuntimeError, "Incorrect value for method_two" 
end

if my_class.method_three != 3
  raise RuntimeError, "Incorrect value for method_three" 
end

if my_class.method_44_4 != 4
  raise RuntimeError, "Incorrect value for method_44_4" 
end

if my_class.predicate_method? != true
  raise RuntimeError, "Incorrect value for predicate_method?" 
end

if my_class.bang_method! != true
  raise RuntimeError, "Incorrect value for bang_method!" 
end
