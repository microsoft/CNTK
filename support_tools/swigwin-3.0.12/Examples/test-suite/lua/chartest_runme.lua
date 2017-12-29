require("import")   -- the import fn
import("chartest")   -- import code

function char_assert(char, code)
  assert(type(char) == 'string')
  assert(char:len() == 1)
  assert(char:byte() == code)
end

char_assert(chartest.GetPrintableChar(), 0x61)
char_assert(chartest.GetUnprintableChar(), 0x7F)

char_assert(chartest.printable_global_char, 0x61)
char_assert(chartest.unprintable_global_char, 0x7F)
