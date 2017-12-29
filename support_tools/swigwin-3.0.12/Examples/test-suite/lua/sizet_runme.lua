require("import")	-- the import fn
import("sizet")	-- import code

s = 2000
s = sizet.test1(s+1)
s = sizet.test2(s+1)
s = sizet.test3(s+1)
s = sizet.test4(s+1)
assert(s == 2004)
