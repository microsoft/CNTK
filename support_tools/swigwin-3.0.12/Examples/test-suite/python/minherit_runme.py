
import minherit

a = minherit.Foo()
b = minherit.Bar()
c = minherit.FooBar()
d = minherit.Spam()

if a.xget() != 1:
    raise RuntimeError, "Bad attribute value"

if b.yget() != 2:
    raise RuntimeError, "Bad attribute value"

if c.xget() != 1 or c.yget() != 2 or c.zget() != 3:
    raise RuntimeError, "Bad attribute value"

if d.xget() != 1 or d.yget() != 2 or d.zget() != 3 or d.wget() != 4:
    raise RuntimeError, "Bad attribute value"


if minherit.xget(a) != 1:
    raise RuntimeError, "Bad attribute value %d" % (minherit.xget(a))

if minherit.yget(b) != 2:
    raise RuntimeError, "Bad attribute value %d" % (minherit.yget(b))

if minherit.xget(c) != 1 or minherit.yget(c) != 2 or minherit.zget(c) != 3:
    raise RuntimeError, "Bad attribute value %d %d %d" % (
        minherit.xget(c), minherit.yget(c), minherit.zget(c))

if minherit.xget(d) != 1 or minherit.yget(d) != 2 or minherit.zget(d) != 3 or minherit.wget(d) != 4:
    raise RuntimeError, "Bad attribute value %d %d %d %d" % (
        minherit.xget(d), minherit.yget(d), minherit.zget(d), minherit.wget(d))

# Cleanse all of the pointers and see what happens

aa = minherit.toFooPtr(a)
bb = minherit.toBarPtr(b)
cc = minherit.toFooBarPtr(c)
dd = minherit.toSpamPtr(d)

if aa.xget() != 1:
    raise RuntimeError, "Bad attribute value"

if bb.yget() != 2:
    raise RuntimeError, "Bad attribute value"

if cc.xget() != 1 or cc.yget() != 2 or cc.zget() != 3:
    raise RuntimeError, "Bad attribute value"

if dd.xget() != 1 or dd.yget() != 2 or dd.zget() != 3 or dd.wget() != 4:
    raise RuntimeError, "Bad attribute value"

if minherit.xget(aa) != 1:
    raise RuntimeError, "Bad attribute value %d" % (minherit.xget(aa))

if minherit.yget(bb) != 2:
    raise RuntimeError, "Bad attribute value %d" % (minherit.yget(bb))

if minherit.xget(cc) != 1 or minherit.yget(cc) != 2 or minherit.zget(cc) != 3:
    raise RuntimeError, "Bad attribute value %d %d %d" % (
        minherit.xget(cc), minherit.yget(cc), minherit.zget(cc))

if minherit.xget(dd) != 1 or minherit.yget(dd) != 2 or minherit.zget(dd) != 3 or minherit.wget(dd) != 4:
    raise RuntimeError, "Bad attribute value %d %d %d %d" % (
        minherit.xget(dd), minherit.yget(dd), minherit.zget(dd), minherit.wget(dd))
