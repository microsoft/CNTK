
# This is the union runtime testcase. It ensures that values within a
# union embedded within a struct can be set and read correctly.

import unions
import sys
import string

# Create new instances of SmallStruct and BigStruct for later use
small = unions.SmallStruct()
small.jill = 200

big = unions.BigStruct()
big.smallstruct = small
big.jack = 300

# Use SmallStruct then BigStruct to setup EmbeddedUnionTest.
# Ensure values in EmbeddedUnionTest are set correctly for each.
eut = unions.EmbeddedUnionTest()

# First check the SmallStruct in EmbeddedUnionTest
eut.number = 1
eut.uni.small = small
Jill1 = eut.uni.small.jill
if (Jill1 != 200):
    print "Runtime test1 failed. eut.uni.small.jill=", Jill1
    sys.exit(1)

Num1 = eut.number
if (Num1 != 1):
    print "Runtime test2 failed. eut.number=", Num1
    sys.exit(1)

# Secondly check the BigStruct in EmbeddedUnionTest
eut.number = 2
eut.uni.big = big
Jack1 = eut.uni.big.jack
if (Jack1 != 300):
    print "Runtime test3 failed. eut.uni.big.jack=", Jack1
    sys.exit(1)

Jill2 = eut.uni.big.smallstruct.jill
if (Jill2 != 200):
    print "Runtime test4 failed. eut.uni.big.smallstruct.jill=", Jill2
    sys.exit(1)

Num2 = eut.number
if (Num2 != 2):
    print "Runtime test5 failed. eut.number=", Num2
    sys.exit(1)
