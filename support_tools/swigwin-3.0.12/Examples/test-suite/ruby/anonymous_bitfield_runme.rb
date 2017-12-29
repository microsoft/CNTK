#!/usr/bin/env ruby
#
# Put script description here.
#
# 
# 
# 
#

require 'swig_assert'
require 'anonymous_bitfield'

include Anonymous_bitfield

foo = Foo.new

{'x' => 4,
  'y' => 3,
  'f' => 1,
  'z' => 8,
  'seq' => 3 }.each do |m, v|
  foo.send("#{m}=", v)
  val = foo.send(m)
  swig_assert("val == v", binding)
end

{'x' => (1 << 4),
  'y' => (1 << 4),
  'f' => (1 << 1),
  'z' => (1 << 16),
  'seq' => (1 << (4*8-6)) }.each do |m, v|
  foo.send("#{m}=", v)
  val = foo.send(m)
  swig_assert("val != v", binding)
end
