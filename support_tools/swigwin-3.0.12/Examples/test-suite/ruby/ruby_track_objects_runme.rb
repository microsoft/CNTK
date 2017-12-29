#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'ruby_track_objects'

def test_same_ruby_object(foo1, foo2)
  if not foo1.equal?(foo2)
    raise "Ruby objects should be the same."
  end
end

def test_same_cpp_object(foo1, foo2)
  if not foo1.cpp_equal(foo2)
    raise "C++ objects should be the same"
  end
end

bar = Ruby_track_objects::Bar.new
foo1 = Ruby_track_objects::Foo.new()
bar.set_unowned_foo(foo1)
  
# test_simple_identity
foo2 = Ruby_track_objects::Foo.new()
foo3 = foo2

test_same_ruby_object(foo2, foo3)
test_same_cpp_object(foo2, foo3)

# test_unowned_foo_identity
foo4 = bar.get_unowned_foo()

test_same_ruby_object(foo1, foo4)
test_same_cpp_object(foo1, foo4)

# test_owned_foo_identity
foo5 = bar.get_owned_foo()
foo6 = bar.get_owned_foo()

test_same_ruby_object(foo5, foo6)
test_same_cpp_object(foo5, foo6)
  
# test_new_foo_identity
foo7 = Ruby_track_objects::Bar.get_new_foo()
foo8 = Ruby_track_objects::Bar.get_new_foo()

if foo7.equal?(foo8)
  raise "Ruby objects should be different."
end

if foo7.cpp_equal(foo8)
  raise "C++ objects should be different."
end
    
# test_set_owned_identity
foo9 = Ruby_track_objects::Foo.new
bar.set_owned_foo(foo9)
foo10 = bar.get_owned_foo()
    
test_same_ruby_object(foo9, foo10)
test_same_cpp_object(foo9, foo10)

# test_set_owned_identity2
begin
  foo11 = Ruby_track_objects::Foo.new
  bar.set_owned_foo(foo11)
  foo11 = nil
end
   
GC.start

foo12 = bar.get_owned_foo()

if not (foo12.say_hello == "Hello")
  raise "Invalid C++ object returned."
end

# test_set_owned_identity3
foo13 = bar.get_owned_foo_by_argument()
foo14 = bar.get_owned_foo_by_argument()

test_same_ruby_object(foo13, foo14)
test_same_cpp_object(foo13, foo14)

# Now create the factory
factory = Ruby_track_objects::Factory.new

# Create itemA which is really an itemB 
itemA = factory.createItem

# Check class
if itemA.class != Ruby_track_objects::ItemA
  raise RuntimeError, 'Item should have an ItemA class'
end

# Now downcast
itemB = Ruby_track_objects.downcast(itemA)

if itemB.class != Ruby_track_objects::ItemB
  raise RuntimeError, 'Item should have an ItemB class'
end

if itemA.eql?(itemB)
  raise RuntimeError, 'Items should be different'
end





