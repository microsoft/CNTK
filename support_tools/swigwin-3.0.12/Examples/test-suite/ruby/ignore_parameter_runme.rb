#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'ignore_parameter'

include Ignore_parameter

# Global function tests
raise RuntimeError unless jaguar(0, 1.0) == "hello"
raise RuntimeError unless lotus("foo", 1.0) == 101
raise RuntimeError unless tvr("foo", 0) == 8.8
raise RuntimeError unless ferrari() == 101

# Member function tests
sc = SportsCars.new
raise RuntimeError unless sc.daimler(0, 1.0) == "hello"
raise RuntimeError unless sc.astonmartin("foo", 1.0) == 101
raise RuntimeError unless sc.bugatti("foo", 0) == 8.8
raise RuntimeError unless sc.lamborghini() == 101

# Constructor tests
MiniCooper.new(0, 1.0)
MorrisMinor.new("foo", 1.0)
FordAnglia.new("foo", 0)
AustinAllegro.new()

