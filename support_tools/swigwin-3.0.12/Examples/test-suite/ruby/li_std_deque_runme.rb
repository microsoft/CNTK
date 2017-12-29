#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'li_std_deque'

include Li_std_deque

# Test constructors for std::deque<int>
intDeque  = IntDeque.new
intDeque2 = IntDeque.new(3)
intDeque3 = IntDeque.new(4, 42)
intDeque4 = IntDeque.new(intDeque3)

# Test constructors for std::deque<double>
doubleDeque  = DoubleDeque.new
doubleDeque2 = DoubleDeque.new(3)
doubleDeque3 = DoubleDeque.new(4, 42.0)
doubleDeque4 = DoubleDeque.new(doubleDeque3)

# Test constructors for std::deque<Real>
realDeque  = RealDeque.new
realDeque2 = RealDeque.new(3)
realDeque3 = RealDeque.new(4, 42.0)
realDeque4 = RealDeque.new(realDeque3)

# average() should return the average of all values in a std::deque<int>
intDeque << 2
intDeque << 4
intDeque << 6
avg = average(intDeque)
raise RuntimeError if avg != 4.0

#
# half() should return a std::deque<float>, where each element is half
# the value of the corresponding element in the input deque<float>.
# The original deque's contents are unchanged.
#
realDeque.clear
realDeque << 2.0
halfDeque = half(realDeque)
raise RuntimeError unless halfDeque[0] == 1.0

#
# halve_in_place() should...
#
halve_in_place(doubleDeque)

