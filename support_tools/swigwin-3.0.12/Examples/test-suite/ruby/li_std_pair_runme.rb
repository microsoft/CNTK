#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'li_std_pair'
include Li_std_pair

swig_assert_each_line(<<'EOF', binding)
#
# Because of template specializations for pair<int, int>, these should return
# an Array of size 2, where both elements are Fixnums.
#
intPair = makeIntPair(7, 6)
intPair.instance_of?(Array)
intPair.size == 2
intPair[0] == 7 && intPair[1] == 6

intPairConstRef = makeIntPairConstRef(7, 6)
intPairConstRef.instance_of?(Array)
intPairConstRef[0] == 7 && intPairConstRef[1] == 6

#
# Each of these should return a reference to a wrapped
# std::pair<int, int> object (i.e. an IntPair instance).
#
intPairPtr = makeIntPairPtr(7, 6)
intPairPtr.instance_of?(IntPair)
intPairPtr[0] == 7 && intPairPtr[1] == 6

intPairRef = makeIntPairRef(7, 6)
intPairRef.instance_of?(IntPair)
intPairRef[0] == 7 && intPairRef[1] == 6
#
# Now test various input typemaps. Each of the wrapped C++ functions
# (product1, product2 and product3) is expecting an argument of a
# different type (see li_std_pair.i). Typemaps should be in place to
# convert this Array into the expected argument type.
#
product1(intPair) == 42
product2(intPair) == 42
product3(intPair) == 42

#
# Similarly, each of the input typemaps should know what to do
# with an IntPair instance.
#
product1(intPairPtr) == 42
product2(intPairPtr) == 42
product3(intPairPtr) == 42
EOF
