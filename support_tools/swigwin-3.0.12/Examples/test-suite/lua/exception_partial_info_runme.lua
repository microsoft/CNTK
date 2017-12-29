require("import")	-- the import fn
import("exception_partial_info")	-- import code

-- catch "undefined" global variables
setmetatable(getfenv(),{__index=function (t,i) error("undefined global variable `"..i.."'",2) end})

imp=exception_partial_info.Impl()

-- trying to call throwing methods
-- should fail
assert(pcall(function() imp:f1() end)==false)
assert(pcall(function() imp:f2() end)==false)
