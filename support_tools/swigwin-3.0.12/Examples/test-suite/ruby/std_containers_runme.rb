#!/usr/bin/env ruby
#
# Standard containers test suite. Tests:
# std::complex, std::vector, std::set and std::map,
# and IN/OUT functions for them.
#
# 
# 
# 
#

require 'swig_assert'
require 'std_containers'
include Std_containers

swig_assert_each_line(<<'EOF', binding)

cube = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]

icube = cident(cube)
icube.each_index { |i| swig_assert_equal("cube[#{i}]","icube[#{i}]", binding, 'cident') }


p = [1,2]
p == pident(p)

v = [1,2,3,4,5,6]
iv = vident(v)
iv.each_index { |i| swig_assert_equal("iv[#{i}]","v[#{i}]", binding, 'vident') }



iv = videntu(v)
iv.each_index { |i| swig_assert_equal("iv[#{i}]","v[#{i}]", binding, 'videntu') }


vu = Vector_ui.new(v)
vu[2] == videntu(vu)[2]

v[0,3][1] == vu[0,3][1]

m = [[1,2,3],[2,3],[3,4]]
im = midenti(m)

0.upto(m.size-1){ |i| 0.upto(m[i].size-1) { |j| swig_assert_equal("m[#{i}][#{j}]","im[#{i}][#{j}]", binding, 'getslice') } }


m = [[1,0,1],[1,1],[1,1]]
im = midentb(m)

0.upto(m.size-1){ |i| 0.upto(m[i].size-1) { |j| swig_assert_equal("(m[#{i}][#{j}]==1)","im[#{i}][#{j}]", binding, 'getslice') } }

mi = Imatrix.new(m)
mc = Cmatrix.new(m)
mi[0][0] == mc[0][0] # or bad matrix

map ={}
map['hello'] = 1
map['hi'] = 2
map['3'] = 2

imap = Std_containers.mapident(map)
map.each_key { |k| swig_assert_equal("map['#{k}']", "imap['#{k}']", binding) }

mapc ={}
c1 = C.new
c2 = C.new
mapc[1] = c1
mapc[2] = c2

mapidentc(mapc)

vi = Vector_i.new([2,2,3,4])
v1 = vi.dup
v1.class == vi.class
v1 != vi
v1.object_id != vi.object_id

v = [1,2]
v1 = v_inout(vi)
vi[1] == v1[1]
# vi.class == v1.class # only if SWIG_RUBY_EXTRA_NATIVE_CONTAINERS was set

v1,v2 = [[1,2],[3,4]]
v1,v2 = v_inout2(v1,v2)
v2 == [1,2]
v1 == [3,4]

a1 = A.new(1)
a2 = A.new(2)

p1 = [1,a1]
p2 = [2,a2]
v  = [p1,p2]
v2 = pia_vident(v)



# v2[0][1].a
# v2[1][1].a

# v3 = Std_containers.vector_piA(v2)

# v3[0][1].a
# v3[1][1].a




s = Set_i.new
s.push(1)
s.push(2)
s.push(3)
j = 1
s.each { |i| swig_assert_equal("#{i}","#{j}", binding, "for s[#{i}]"); j += 1 }


EOF

