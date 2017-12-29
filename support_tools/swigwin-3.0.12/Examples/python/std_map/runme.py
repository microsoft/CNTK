# file: runme.py

import example

pmap = example.pymap()
pmap["hi"] = 1
pmap["hello"] = 2


dmap = {}
dmap["hello"] = 1.0
dmap["hi"] = 2.0

print dmap.items()
print dmap.keys()
print dmap.values()

print dmap
hmap = example.halfd(dmap)
dmap = hmap

print dmap
for i in dmap.iterkeys():
    print "key", i

for i in dmap.itervalues():
    print "val", i

for k, v in dmap.iteritems():
    print "item", k, v

dmap = example.DoubleMap()
dmap["hello"] = 1.0
dmap["hi"] = 2.0

for i in dmap.iterkeys():
    print "key", i

for i in dmap.itervalues():
    print "val", i

for k, v in dmap.iteritems():
    print "item", k, v


print dmap.items()
print dmap.keys()
print dmap.values()

hmap = example.halfd(dmap)
print hmap.keys()
print hmap.values()


dmap = {}
dmap["hello"] = 2
dmap["hi"] = 4

hmap = example.halfi(dmap)
print hmap
print hmap.keys()
print hmap.values()


dmap = hmap

for i in dmap.iterkeys():
    print "key", i

for i in dmap.itervalues():
    print "val", i

for i in dmap.iteritems():
    print "item", i

for k, v in dmap.iteritems():
    print "item", k, v

print dmap
