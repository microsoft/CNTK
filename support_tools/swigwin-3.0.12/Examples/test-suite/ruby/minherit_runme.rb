#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'minherit'

a = Minherit::Foo.new
b = Minherit::Bar.new
c = Minherit::FooBar.new
d = Minherit::Spam.new

if a.xget() != 1
  raise RuntimeError, "Bad attribute value"
end

if b.yget() != 2
  raise RuntimeError, "Bad attribute value"
end    

if c.xget() != 1 or c.yget() != 2 or c.zget() != 3
  raise RuntimeError, "Bad attribute value"
end

if d.xget() != 1 or d.yget() != 2 or d.zget() != 3 or d.wget() != 4
  raise RuntimeError, "Bad attribute value"
end

if Minherit.xget(a) != 1
  raise RuntimeError, "Bad attribute value #{Minherit.xget(a)}"
end     

if Minherit.yget(b) != 2
  raise RuntimeError, "Bad attribute value #{Minherit.yget(b)}"
end

if Minherit.xget(c) != 1 or Minherit.yget(c) != 2 or Minherit.zget(c) != 3
  raise RuntimeError, "Bad attribute value #{Minherit.xget(c)} #{Minherit.yget(c)} #{Minherit.zget(c)}"
end    

if Minherit.xget(d) != 1 or Minherit.yget(d) != 2 or Minherit.zget(d) != 3 or Minherit.wget(d) != 4
  raise RuntimeError, "Bad attribute value #{Minherit.xget(d)} #{Minherit.yget(d)} #{Minherit.zget(d)} #{Minherit.wget(d)}"
end

# Cleanse all of the pointers and see what happens

aa = Minherit.toFooPtr(a)
bb = Minherit.toBarPtr(b)
cc = Minherit.toFooBarPtr(c)
dd = Minherit.toSpamPtr(d)

if aa.xget() != 1
  raise RuntimeError, "Bad attribute value"
end

if bb.yget() != 2
  raise RuntimeError, "Bad attribute value"
end

if cc.xget() != 1 or cc.yget() != 2 or cc.zget() != 3
  raise RuntimeError, "Bad attribute value"
end

if dd.xget() != 1 or dd.yget() != 2 or dd.zget() != 3 or dd.wget() != 4
  raise RuntimeError, "Bad attribute value"
end

if Minherit.xget(aa) != 1
  raise RuntimeError, "Bad attribute value #{Minherit.xget(aa)}"
end

if Minherit.yget(bb) != 2
  raise RuntimeError, "Bad attribute value #{Minherit.yget(bb)}"
end

if Minherit.xget(cc) != 1 or Minherit.yget(cc) != 2 or Minherit.zget(cc) != 3
  raise RuntimeError, "Bad attribute value #{Minherit.xget(cc)} #{Minherit.yget(cc)} #{Minherit.zget(cc)}"
end

if Minherit.xget(dd) != 1 or Minherit.yget(dd) != 2 or Minherit.zget(dd) != 3 or Minherit.wget(dd) != 4
  raise RuntimeError, "Bad attribute value #{Minherit.xget(dd)} #{Minherit.yget(dd)} #{Minherit.zget(dd)} #{Minherit.wget(dd)}"
end

