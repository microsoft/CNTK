require("import")	-- the import fn
import("char_strings")	-- import code

assert (char_strings.CharPingPong("hi there") == "hi there")
assert (char_strings.CharPingPong(nil) == nil)

assert (char_strings.CharArrayPingPong("hi there") == "hi there")
assert (char_strings.CharArrayPingPong(nil) == nil)

assert (char_strings.CharArrayDimsPingPong("hi there") == "hi there")
assert (char_strings.CharArrayDimsPingPong(nil) == nil)

