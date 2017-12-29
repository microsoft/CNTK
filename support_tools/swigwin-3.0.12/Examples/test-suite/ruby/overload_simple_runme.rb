#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'overload_simple'

include Overload_simple

if foo(3) != "foo:int"
  raise RuntimeError, "foo(int)"
end

if foo(3.0) != "foo:double"
  raise RuntimeError, "foo(double)"
end

if foo("hello") != "foo:char *"
  raise RuntimeError, "foo(char *)"
end

f = Foo.new
b = Bar.new

if foo(f) != "foo:Foo *"
  raise RuntimeError, "foo(Foo *)"
end

if foo(b) != "foo:Bar *"
  raise RuntimeError, "foo(Bar *)"
end

v = malloc_void(32)

if foo(v) != "foo:void *"
  raise RuntimeError, "foo(void *)"
end

s = Spam.new

if s.foo(3) != "foo:int"
  raise RuntimeError, "Spam::foo(int)"
end

if s.foo(3.0) != "foo:double"
  raise RuntimeError, "Spam::foo(double)"
end

if s.foo("hello") != "foo:char *"
  raise RuntimeError, "Spam::foo(char *)"
end

if s.foo(f) != "foo:Foo *"
  raise RuntimeError, "Spam::foo(Foo *)"
end

if s.foo(b) != "foo:Bar *"
  raise RuntimeError, "Spam::foo(Bar *)"
end

if s.foo(v) != "foo:void *"
  raise RuntimeError, "Spam::foo(void *)"
end

if Spam.bar(3) != "bar:int"
  raise RuntimeError, "Spam::bar(int)"
end

if Spam.bar(3.0) != "bar:double"
  raise RuntimeError, "Spam::bar(double)"
end

if Spam.bar("hello") != "bar:char *"
  raise RuntimeError, "Spam::bar(char *)"
end

if Spam.bar(f) != "bar:Foo *"
  raise RuntimeError, "Spam::bar(Foo *)"
end

if Spam.bar(b) != "bar:Bar *"
  raise RuntimeError, "Spam::bar(Bar *)"
end

if Spam.bar(v) != "bar:void *"
  raise RuntimeError, "Spam::bar(void *)"
end

# Test constructors

s = Spam.new
if s.type != "none"
  raise RuntimeError, "Spam()"
end

s = Spam.new(3)
if s.type != "int"
  raise RuntimeError, "Spam(int)"
end
    
s = Spam.new(3.4)
if s.type != "double"
  raise RuntimeError, "Spam(double)"
end

s = Spam.new("hello")
if s.type != "char *"
  raise RuntimeError, "Spam(char *)"
end

s = Spam.new(f)
if s.type != "Foo *"
  raise RuntimeError, "Spam(Foo *)"
end

s = Spam.new(b)
if s.type != "Bar *"
  raise RuntimeError, "Spam(Bar *)"
end

s = Spam.new(v)
if s.type != "void *"
  raise RuntimeError, "Spam(void *)"
end
