#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'default_constructor'

include Default_constructor

# Ruby 1.6 raises NameError if you try to call Class.new where no constructor
# is defined; Ruby 1.7 changed this to NoMethodError

NoConstructorError = Kernel.const_defined?("NoMethodError") ? NoMethodError : NameError

# This should be no problem
a = A.new

# Nor should this
aa = AA.new

# The default constructor for B is private, so this should raise an exception
begin
  b = B.new
rescue ArgumentError
  # pass
rescue TypeError
  # In Ruby 1.8 the exception raised is:
  # TypeError: allocator undefined for Default_constructor::BB
	exceptionRaised = true
end

# The two-argument constructor for B should work
b = B.new(3, 4)

# BB shouldn't inherit B's default constructor, so this should raise an exception
begin
  bb = BB.new
  puts "Whoa. new BB created."
rescue NoConstructorError
  # pass
rescue TypeError
  # In Ruby 1.8 the exception raised is:
  # TypeError: allocator undefined for Default_constructor::BB
	exceptionRaised = true
end

# C's constructor is protected, so this should raise an exception
begin
  c = C.new
  print "Whoa. new C created."
rescue NoConstructorError
  # pass
rescue TypeError
  # In Ruby 1.8 the exception raised is:
  # TypeError: allocator undefined for Default_constructor::C
  # pass
rescue TypeError
  # In Ruby 1.8 the exception raised is:
  # TypeError: allocator undefined for Default_constructor::C
  # pass
end

# CC gets a default constructor, so no problem here
cc = CC.new

# D's constructor is private, so this should fail
begin
  d = D.new
  puts "Whoa. new D created"
rescue NoConstructorError
  # pass
rescue TypeError
  # In Ruby 1.8 the exception raised is:
  # TypeError: allocator undefined for Default_constructor::D
  # pass
end

# DD shouldn't get a default constructor, so this should fail
begin
  dd = DD.new
  puts "Whoa. new DD created"
rescue NoConstructorError
  # pass
rescue TypeError
  # In Ruby 1.8 the exception raised is:
  # TypeError: allocator undefined for Default_constructor::DD
  # pass
rescue TypeError
  # In Ruby 1.8 the exception raised is:
  # TypeError: allocator undefined for Default_constructor::D
  # pass
rescue TypeError
  # In Ruby 1.8 the exception raised is:
  # TypeError: allocator undefined for Default_constructor::DD
  # pass
end

# AD shouldn't get a default constructor, so this should fail
begin
  ad = AD.new
  puts "Whoa. new AD created"
rescue NoConstructorError
  # pass
rescue TypeError
  # In Ruby 1.8 the exception raised is:
  # TypeError: allocator undefined for Default_constructor::AD
  # pass
rescue TypeError
  # In Ruby 1.8 the exception raised is:
  # TypeError: allocator undefined for Default_constructor::AD
  # pass
end

# Both of the arguments to E's constructor have default values,
# so this should be fine.
e = E.new

# EE should get a default constructor
ee = EE.new

# EB should not get a default constructor (because B doesn't have one)
begin
  eb = EB.new
  puts "Whoa. new EB created"
rescue NoConstructorError
  # pass
rescue TypeError
  # In Ruby 1.8 the exception raised is:
  # TypeError: allocator undefined for Default_constructor::EB
  # pass
rescue TypeError
  # In Ruby 1.8 the exception raised is:
  # TypeError: allocator undefined for Default_constructor::EB
  # pass
end

# This should work fine
f = F.new

# This should work fine
g = G.new

# This should work fine
gg = GG.new
