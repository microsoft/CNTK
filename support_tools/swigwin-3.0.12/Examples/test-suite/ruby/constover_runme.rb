#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'constover'

error = 0

p = Constover.test("test")
if p != "test"
  puts "test failed!"
  error = 1
end

p = Constover.test_pconst("test")
if p != "test_pconst"
  puts "test_pconst failed!"
  error = 1
end
    
f = Constover::Foo.new
p = f.test("test")
if p != "test"
  print "member-test failed!"
  error = 1
end

p = f.test_pconst("test")
if p != "test_pconst"
  print "member-test_pconst failed!"
  error = 1
end

p = f.test_constm("test")
if p != "test_constmethod"
  print "member-test_constm failed!"
  error = 1
end

p = f.test_pconstm("test")
if p != "test_pconstmethod"
  print "member-test_pconstm failed!"
  error = 1
end
    
exit(error)


