require("import")	-- the import fn
import("ret_by_value")	-- import code

a = ret_by_value.get_test()
assert(a.myInt == 100)
assert(a.myShort == 200)
