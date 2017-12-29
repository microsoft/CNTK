#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'enum_thorough'

include Enum_thorough

# Just test an in and out typemap for enum SWIGTYPE and const enum SWIGTYPE & typemaps
raise RuntimeError if speedTest4(SpeedClass::Slow) != SpeedClass::Slow
raise RuntimeError if speedTest5(SpeedClass::Slow) != SpeedClass::Slow

