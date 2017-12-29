#!/usr/bin/env ruby
#
# This is a test of STL containers, iterators and using proc
# objects to change the sorting function used in them.  Same as a
# std::binary_predicate in C++.
#
# 
# 
# 
#

require 'swig_assert'
require 'stl_new'


def _sequence(container)
  swig_assert_each_line(<<'EOF', binding)
cont = container.new([9,1,8,2,7,3,6,4,5])
cont.to_a == [9,1,8,2,7,3,6,4,5]
cont.size == 9
i = cont.begin
i.class == Stl_new::Iterator
cont.end - cont.begin == cont.size
cont.begin.value == 9
(cont.end-1).value == 5
cont[0],cont[1] = cont[1],cont[0]
cont.to_a == [1,9,8,2,7,3,6,4,5]
i0 = cont.begin
i1 = i0+1
tmp = i0.value   # tmp = 1
tmp == 1
i0.value = i1.value # elem[0] = 9
i1.value = tmp      # elem[1] = 1
cont.to_a == [9,1,8,2,7,3,6,4,5]
i0 += 8
prev = i0.value
i0 -= 8
cur = i0.value
i0.value = prev
prev = cur
i0 += 8
cur = i0.value
i0.value = prev
cont.to_a == [5,1,8,2,7,3,6,4,9]
i0 == cont.end-1
i0 != cont.end
EOF
end

def _random_iterator(container)
  swig_assert_each_line(<<EOF, binding)
  cont = #{container}.new([9,1,8,2,7,3,6,4,5])
  Stl_new.nth_element(cont.begin,cont.begin+cont.size/2,cont.end)
  cont.to_a == [3, 1, 2, 4, 5, 6, 7, 8, 9]
  Stl_new.nth_element(cont.begin,cont.begin+1,cont.end, proc { |a,b| b<a } )
  cont.to_a == [9, 8, 7, 6, 5, 4, 2, 1, 3]
EOF
end

def _set(container)
  swig_assert_each_line(<<EOF, binding)
    cont = #{container}.new
    [9,1,8,2,7,3,6,4,5].each { |x| cont.insert(x) }
    i0 = cont.begin()
    cont.to_a == [1,2,3,4,5,6,7,8,9]
    cont = #{container}.new( proc { |a,b| b < a } )
    [9,1,8,2,7,3,6,4,5].each { |x| cont.insert(x) }
    cont.to_a == [9, 8, 7, 6, 5, 4, 3, 2, 1]
    cont = #{container}.new( proc { |a,b| b > a } )
    [9,1,8,2,7,3,6,4,5].each { |x| cont.insert(x) }
    cont.to_a == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    cont = #{container}.new(proc { |a,b| b < a } )
    cont.insert(1)
    cont.to_a == [1]
    i0 = cont.begin()
    cont.erase(i0) # don't use i0 anymore, it is invalid now
    cont.to_a == []
EOF
end
    
def _map(container)
  swig_assert_each_line(<<EOF, binding)
    cont = #{container}.new
    cont['z'] = 9
    cont['y'] = 1
    cont['x'] = 8
    cont['w'] = 2
    cont.to_a == [['w',2],['x',8],['y',1],['z',9]]

    cont = #{container}.new(proc { |a,b| b < a } )
    cont['z'] = 9
    cont['y'] = 1
    cont['x'] = 8
    cont['w'] = 2
    cont.to_a == [['z',9],['y',1],['x',8],['w',2]]

    #cont.iterator.to_a == [['w',2],['x',8],['y',1],['z',9]]
EOF
end

def test
  for container in [Stl_new::Vector, Stl_new::Deque, Stl_new::List]
    yield method(:_sequence), container
  end
  yield method(:_set), Stl_new::Set
  yield method(:_map), Stl_new::Map
#   for container in [Stl_new::Vector, Stl_new::Deque]
#     yield method(:_random_iterator), container
#   end
end


test do |proc, container|
  proc.call(container)
end


