#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'cpp11_li_std_array'

include Cpp11_li_std_array


def failed(a, b, msg)
    raise RuntimeError, "#{msg} #{a} #{b}"
end

def compare_sequences(a, b)
    if a.size != b.size
        failed(a, b, "different sizes")
    end
    for i in 0..a.size-1
      failed(a, b, "elements are different:") if a[i] != b[i]
    end
end

def compare_containers(rubyarray, swigarray)
    compare_sequences(rubyarray, swigarray)
end

def setslice_exception(swigarray, newval)
    begin
        swigarray[0..swigarray.size] = newval
        raise RuntimeError, "swigarray[] = #{newval} missed set exception for swigarray: #{swigarray}"
    rescue ArgumentError => e
#        print "exception: #{e}"
    end
end


# Check std::array has similar behaviour to a Ruby array
# except it is not resizable

ps = [0, 1, 2, 3, 4, 5]

ai = ArrayInt6.new(ps)

compare_containers(ps, ai)

# slices
compare_containers(ps[0..5], ai[0..5])
compare_containers(ps[-6..-1], ai[-6..-1])
compare_containers(ps[0..10], ai[0..10])

# Reverse (.reverse is not provided)
rev = []
ai.reverse_each { |i| rev.push i }
compare_containers(ps.reverse, rev)

# Modify content
for i in 0..ps.size-1
    ps[i] = ps[i] * 10
    ai[i] = ai[i] * 10
end
compare_containers(ps, ai)

# Empty
ai = ArrayInt6.new()
compare_containers([0, 0, 0, 0, 0, 0], ai)

# Set slice
newvals = [10, 20, 30, 40, 50, 60]
ai[0, 6] = newvals
compare_containers(ai, newvals)

ai[-6, 6] = newvals
compare_containers(ai, newvals)

setslice_exception(ai, [1, 2, 3, 4, 5, 6, 7])
setslice_exception(ai, [1, 2, 3, 4, 5])
setslice_exception(ai, [1, 2, 3, 4])
setslice_exception(ai, [1, 2, 3])
setslice_exception(ai, [1, 2])
setslice_exception(ai, [1])
setslice_exception(ai, [])

# Check return
compare_containers(arrayOutVal(), [-2, -1, 0, 0, 1, 2])
compare_containers(arrayOutConstRef(), [-2, -1, 0, 0, 1, 2])
compare_containers(arrayOutRef(), [-2, -1, 0, 0, 1, 2])
compare_containers(arrayOutPtr(), [-2, -1, 0, 0, 1, 2])

# Check passing arguments
ai = arrayInVal([9, 8, 7, 6, 5, 4])
compare_containers(ai, [90, 80, 70, 60, 50, 40])

ai = arrayInConstRef([9, 8, 7, 6, 5, 4])
compare_containers(ai, [90, 80, 70, 60, 50, 40])

ai = ArrayInt6.new([9, 8, 7, 6, 5, 4])
arrayInRef(ai)
compare_containers(ai, [90, 80, 70, 60, 50, 40])

ai = ArrayInt6.new([9, 8, 7, 6, 5, 4])
arrayInPtr(ai)
compare_containers(ai, [90, 80, 70, 60, 50, 40])

# indexing
ai = ArrayInt6.new([9, 8, 7, 6, 5, 4])
swig_assert_equal(ai[0], 9, binding)
swig_assert_equal(ai[5], 4, binding)
swig_assert_equal(ai[6], nil, binding)
swig_assert_equal(ai[-7], nil, binding)

# fill
ai.fill(111)
compare_containers(ai, [111, 111, 111, 111, 111, 111])

# various
ai = ArrayInt6.new([9, 8, 7, 6, 5, 4])
swig_assert(ai.include? 9)
swig_assert(!ai.include?(99))
swig_assert(ai.kind_of? ArrayInt6)
swig_assert(ai.find {|x| x == 6 } == 6)
swig_assert(ai.find {|x| x == 66 } == nil)
swig_assert(ai.respond_to?(:each))
swig_assert(ai.respond_to?(:each_with_index))

ai = [0, 10, 20, 30, 40, 50]
ai.each_with_index { |e,i| swig_assert(e/10 == i) }

