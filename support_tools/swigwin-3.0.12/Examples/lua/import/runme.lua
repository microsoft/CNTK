# Test various properties of classes defined in separate modules

print("Testing the %import directive")

if string.sub(_VERSION,1,7)=='Lua 5.0' then
	-- lua5.0 doesnt have a nice way to do this
	function loadit(a)
		lib=loadlib(a..'.dll','luaopen_'..a) or loadlib(a..'.so','luaopen_'..a)
		assert(lib)()
	end
	loadit('base')
	loadit('foo')
	loadit('bar')
	loadit('spam')
else
	-- lua 5.1 does
	require 'base'
	require 'foo'
	require 'bar'
	require 'spam'
end

-- Create some objects

print("Creating some objects")

a = base.Base()
b = foo.Foo()
c = bar.Bar()
d = spam.Spam()

-- Try calling some methods
print("Testing some methods")
print("Should see 'Base::A' ---> ",a:A())
print("Should see 'Base::B' ---> ",a:B())

print("Should see 'Foo::A' ---> ",b:A())
print("Should see 'Foo::B' ---> ",b:B())

print("Should see 'Bar::A' ---> ",c:A())
print("Should see 'Bar::B' ---> ",c:B())

print("Should see 'Spam::A' ---> ",d:A())
print("Should see 'Spam::B' ---> ",d:B())

-- Try some casts

print("\nTesting some casts")

x = a:toBase()
print("Should see 'Base::A' ---> ",x:A())
print("Should see 'Base::B' ---> ",x:B())

x = b:toBase()
print("Should see 'Foo::A' ---> ",x:A())
print("Should see 'Base::B' ---> ",x:B())

x = c:toBase()
print("Should see 'Bar::A' ---> ",x:A())
print("Should see 'Base::B' ---> ",x:B())

x = d:toBase()
print("Should see 'Spam::A' ---> ",x:A())
print("Should see 'Base::B' ---> ",x:B())

x = d:toBar()
print("Should see 'Bar::B' ---> ",x:B())


print "\nTesting some dynamic casts\n"
x = d:toBase()

print " Spam -> Base -> Foo : "
y = foo.Foo_fromBase(x)
if y then
      print "bad swig"
else
      print "good swig"
end

print " Spam -> Base -> Bar : "
y = bar.Bar_fromBase(x)
if y then
      print "good swig"
else
      print "bad swig"
end
      
print " Spam -> Base -> Spam : "
y = spam.Spam_fromBase(x)
if y then
      print "good swig"
else
      print "bad swig"
end

print " Foo -> Spam : "
y = spam.Spam_fromBase(b)
if y then
      print "bad swig"
else
      print "good swig"
end
