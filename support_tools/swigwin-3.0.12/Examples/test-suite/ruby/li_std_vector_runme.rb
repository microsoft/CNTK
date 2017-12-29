#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'li_std_vector'

include Li_std_vector

iv = IntVector.new(4)

swig_assert( "iv.respond_to? :each", binding )

begin
  iv.each
  swig_assert( false, nil, "iv.each worked with no block!")
rescue ArgumentError
end

swig_assert_each_line(<<'EOF', binding)
iv.respond_to?(:each) == true
iv.respond_to?(:each_with_index) == true
EOF

iv.each_with_index { |e,i| 
  swig_assert("#{e} == 0", binding, "for iv[#{i}] == 0") 
}

0.upto(3) { |i| iv[i] = i }

{ "iv[-1]" => 3,
  "iv.slice(0,2).to_s" => "01", 
  "iv.slice(1,2).to_s" => "12", 
  "iv[0,-2]" => nil,
  "iv[0,3].to_s" => "012",
  "iv[0,10].to_s" => "0123",
  "iv[1..2].to_s" => '12',
  "iv[1..3].to_s" => '123',
  "iv[1..4].to_s" => '123',
  "iv[1..-2].to_s" => '12',
  "iv[2..-3].to_s" => '',
}.each do |k,v|
  swig_assert( "#{k} == #{v.inspect}", binding )
end

swig_assert_each_line(<<'EOF', binding)
iv << 5
iv.push 5
iv.pop == 5
iv.unshift(7)
iv.shift == 7
iv.unshift(7, 3)
iv.insert(1,5)
iv.insert(0, 3)
iv.unshift(2,3,4)
x = average(iv)
y = average([1, 2, 3, 4])
half([10, 10.5, 11, 11.5])
EOF

iv = IntVector.new([0,1,2,3,4,5,6])
iv.delete_if { |x| x == 0 || x == 3 || x == 6 }
swig_assert_equal(iv.to_s, '1245', binding)

iv[1,2] = [-2, -4]
swig_assert_equal(iv.to_s, '1-2-45', binding)

iv = IntVector.new([0,1,2,3])
iv[0,1] = [-1, -2]
swig_assert_equal(iv.to_s, '-1-2123', binding)

iv = IntVector.new([1,2,3,4])
iv[1,3] = [6,7,8,9]
#__setitem__ needs fixing
#swig_assert_equal(iv.to_s, '16789', binding)

iv = IntVector.new([1,2,3,4])

swig_assert_equal(iv[0], 1, binding)
swig_assert_equal(iv[3], 4, binding)
swig_assert_equal(iv[4], nil, binding)
swig_assert_equal(iv[-5], nil, binding)

iv[-1] = 9
iv[-4] = 6
swig_assert_equal(iv.to_s, '6239', binding)

begin
  iv[-5] = 99
  raise "exception missed"
rescue IndexError
end

iv[6] = 5
swig_assert_equal(iv.to_s, '6239555', binding)

def failed(a, b, msg)
    a = 'nil' if a == nil
    b = 'nil' if b == nil
    raise RuntimeError, "#{msg}: #{a} ... #{b}"
end

def compare_sequences(a, b)
    if a != nil && b != nil
      if a.size != b.size
        failed(a, b, "different sizes")
      end
      for i in 0..a.size-1
        failed(a, b, "elements are different") if a[i] != b[i]
      end
    else
      unless a == nil && b == nil
        failed(a, b, "only one of the sequences is nil")
      end
    end
end

def compare_expanded_sequences(a, b)
    # a can contain nil elements which indicate additional elements
    # b won't contain nil for additional elements
    if a != nil && b != nil
      if a.size != b.size
        failed(a, b, "different sizes")
      end
      for i in 0..a.size-1
        failed(a, b, "elements are different") if a[i] != b[i] && a[i] != nil
      end
    else
      unless a == nil && b == nil
        failed(a, b, "only one of the sequences is nil")
      end
    end
end

def check_slice(i, length)
  aa = [1,2,3,4]
  iv = IntVector.new(aa)

  aa_slice = aa[i, length]
  iv_slice = iv[i, length]
  compare_sequences(aa_slice, iv_slice)

  aa_slice = aa.slice(i, length)
  iv_slice = iv.slice(i, length)
  compare_sequences(aa_slice, iv_slice)
end

def check_range(i, j)
  aa = [1,2,3,4]
  iv = IntVector.new(aa)

  aa_range = aa[i..j]
  iv_range = iv[i..j]
  compare_sequences(aa_range, iv_range)

  aa_range = aa[Range.new(i, j, true)]
  iv_range = iv[Range.new(i, j, true)]
  compare_sequences(aa_range, iv_range)
end

def set_slice(i, length, expect_nil_expanded_elements)
  aa = [1,2,3,4]
  iv = IntVector.new(aa)
  aa_new = [8, 9]
  iv_new = IntVector.new(aa_new)

  aa[i, length] = aa_new
  iv[i, length] = iv_new
  if expect_nil_expanded_elements
    compare_expanded_sequences(aa, iv)
  else
    compare_sequences(aa, iv)
  end
end

for i in -5..5
  for length in -5..5
    check_slice(i, length)
  end
end

for i in -5..5
  for j in -5..5
    check_range(i, j)
  end
end

for i in -4..4
  for length in 0..4
    set_slice(i, length, false)
  end
end

for i in [5, 6]
  for length in 0..5
    set_slice(i, length, true)
  end
end


dv = DoubleVector.new(10)

swig_assert( "dv.respond_to? :each_with_index", binding )

dv.each_with_index { |e,i| swig_assert_equal("dv[#{i}]", 0.0, binding) }

0.upto(9) { |i| dv[i] = i/2.0 }

{ "dv[-1]" => 4.5,
  "dv.slice(0,2).to_s" => "0.00.5",
  "dv[0,-2]" => nil,
  "dv[0,3].to_s" => "0.00.51.0",
  "dv[3,3].to_s" => "1.52.02.5",
}.each do |k,v|
  swig_assert_equal( k, v.inspect, binding )
end

swig_assert_each_line(<<'EOF', binding)
dv.delete_at(2)
dv.delete_if { |x| x == 2.0 }
dv.include? 3.0
dv.find {|x| x == 3.0 }
dv.kind_of? DoubleVector
halved = []
halved = dv.map { |x| x / 2 }
halve_in_place(dv)
halved.to_a == dv.to_a
sv = StructVector.new
sv << Li_std_vector::Struct.new
sv[0].class == Li_std_vector::Struct
sv[1] = Li_std_vector::Struct.new

EOF

bv = BoolVector.new(2)
[true, false, true, true].each { |i| bv.push(i) }
0.upto(bv.size-1) { |i| bv[i] = !bv[i] }
bv_check = [true, true, false, true, false, false]
for i in 0..bv.size-1 do
  swig_assert(bv_check[i] == bv[i], binding, "bv[#{i}]")
end

swig_assert_each_line(<<'EOF', binding)
lv = LanguageVector.new
lv << 1
lv << [1,2]
lv << 'asd'
lv[0], lv[1] = lv[1], lv[0]
EOF


# this should assert
begin
  lv = LanguageVector.new('crapola')
rescue
end
