#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'director_wombat'

include Director_wombat

# Test base class functionality
barObj = Bar.new

# Bar#meth should return a Foo_integers instance
fooIntsObj = barObj.meth
raise RuntimeError unless fooIntsObj.instance_of?(Foo_integers)

# Foo_integers#meth(n) should return n
raise RuntimeError if fooIntsObj.meth(42) != 42

#
# Now subclass Foo_integers, but override its virtual method
# meth(n) so that it returns the number plus one.
#
class MyFooInts < Foo_integers
  def meth(n)
    n + 1
  end
end

#
# Subclass Bar and override its virtual method meth()
# so that it returns a new MyFooInts instance instead of
# a Foo_integers instance.
#
class MyBar < Bar
  def meth
    MyFooInts.new
  end
end

#
# Now repeat previous tests:
#
# Create a MyBar instance...
#
barObj = MyBar.new

# MyBar#meth should return a MyFooInts instance
fooIntsObj = barObj.meth
raise RuntimeError unless fooIntsObj.instance_of?(MyFooInts)

# MyFooInts#meth(n) should return n+1
raise RuntimeError if fooIntsObj.meth(42) != 43

