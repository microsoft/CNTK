require("import")	-- the import fn
import("li_std_except")	-- import code

test = li_std_except.Test()
-- under lua, all the std::exceptions are just turned to strings, so we are only checking that is fails
assert(pcall(function() test:throw_bad_exception() end)==false)
assert(pcall(function() test:throw_domain_error() end)==false)
assert(pcall(function() test:throw_exception() end)==false)
assert(pcall(function() test:throw_invalid_argument() end)==false)
assert(pcall(function() test:throw_length_error() end)==false)
assert(pcall(function() test:throw_logic_error() end)==false)
assert(pcall(function() test:throw_out_of_range() end)==false)
assert(pcall(function() test:throw_overflow_error() end)==false)
assert(pcall(function() test:throw_range_error() end)==false)
assert(pcall(function() test:throw_runtime_error() end)==false)
assert(pcall(function() test:throw_underflow_error() end)==false)
