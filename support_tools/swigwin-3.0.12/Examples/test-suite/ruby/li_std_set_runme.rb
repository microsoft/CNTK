#!/usr/bin/env ruby
#
# Put script description here.
#
# 
# 
# 
#

require 'swig_assert'

require 'li_std_set'
include Li_std_set

swig_assert_each_line(<<'EOF', binding)
s = Set_string.new

s.push("a")
s.push("b")
s << "c"

sum = ''
s.each { |x| sum << x }
sum == 'abc'

b = s.begin  # only if swig iterators are on
e = s.end
sum = ''
while b != e; sum << b.value; b.next; end
sum == 'abc'

b = s.rbegin  # only if swig iterators are on
e = s.rend
sum = ''
while b != e; sum << b.value; b.next; end
sum == 'cba'


si = Set_int.new
si << 1
si.push(2)
si.push(3)

i = s.begin()
i.next()
s.erase(i)
s.to_s == 'ac'

b = s.begin
e = s.end
e - b == 2

m = b + 1
m.value == 'c'

s = LanguageSet.new
s.insert([1,2])
s.insert(1)
s.insert("hello")
#s.to_a == [1,[1,2],'hello']  # sort order: s.sort {|a,b| a.hash <=> b.hash}
# Test above is flawed as LanguageSet sorts by each element's hash, so the order will change from one invocation to the next. Sort a conversion to array instead.
sa = s.to_a.sort { |x, y| x.to_s <=> y.to_s }
sa == [1,[1,2],'hello']

EOF

iv = Set_int.new([0,1,2,3,4,5,6])
iv.delete_if { |x| x == 0 || x == 3 || x == 6 }
swig_assert_equal(iv.to_s, '1245', binding)

