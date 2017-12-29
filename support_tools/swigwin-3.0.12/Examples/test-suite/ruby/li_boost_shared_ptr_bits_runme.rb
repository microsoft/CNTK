require 'li_boost_shared_ptr_bits'
require 'swig_gc'

v = Li_boost_shared_ptr_bits::VectorIntHolder.new()
v.push(Li_boost_shared_ptr_bits::IntHolder.new(11))
v.push(Li_boost_shared_ptr_bits::IntHolder.new(22))
v.push(Li_boost_shared_ptr_bits::IntHolder.new(33))

sum = Li_boost_shared_ptr_bits::sum(v)
if (sum != 66)
  raise RuntimeError, "sum is wrong"
end

hidden = Li_boost_shared_ptr_bits::HiddenDestructor.create()
GC.track_class = Li_boost_shared_ptr_bits::HiddenPrivateDestructor
GC.stats if $VERBOSE
hidden = nil
GC.start

hiddenPrivate = Li_boost_shared_ptr_bits::HiddenPrivateDestructor.create()
if (Li_boost_shared_ptr_bits::HiddenPrivateDestructor.DeleteCount != 0)
  # GC doesn't always run
#  raise RuntimeError, "Count should be zero"
end

GC.stats if $VERBOSE
hiddenPrivate = nil
GC.start
if (Li_boost_shared_ptr_bits::HiddenPrivateDestructor.DeleteCount != 1)
  # GC doesn't always run
#  raise RuntimeError, "Count should be one"
end
