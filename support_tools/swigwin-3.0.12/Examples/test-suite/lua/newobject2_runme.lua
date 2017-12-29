require("import")	-- the import fn
import("newobject2",true)	-- import code

foo1 = newobject2.makeFoo()	-- lua doesnt yet support static fns properly
assert(newobject2.fooCount() == 1)	-- lua doesnt yet support static fns properly

foo2 = newobject2.makeFoo()
assert(newobject2.fooCount() == 2)

foo1 = nil 
collectgarbage()
assert(newobject2.fooCount() == 1)

foo2 = nil 
collectgarbage()
assert(newobject2.fooCount() == 0)
