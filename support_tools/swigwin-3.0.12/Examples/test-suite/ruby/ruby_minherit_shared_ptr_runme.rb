#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'ruby_minherit_shared_ptr'

md = Ruby_minherit_shared_ptr::MultiDerived.new(11, 22)

if md.Base1Func != 11 then
 raise RuntimeError
end
if md.Interface1Func != 22 then
 raise RuntimeError
end
if Ruby_minherit_shared_ptr.BaseCheck(md) != 11 then
 raise RuntimeError
end
if Ruby_minherit_shared_ptr.InterfaceCheck(md) != 22 then
 raise RuntimeError
end
if Ruby_minherit_shared_ptr.DerivedCheck(md) != 33 then
 raise RuntimeError
end
