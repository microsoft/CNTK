import cpp11_alternate_function_syntax

a = cpp11_alternate_function_syntax.SomeStruct()

res = a.addNormal(4, 5)
if res != 9:
    raise RuntimeError, ("SomeStruct::addNormal(4,5) returns ",
                         res, " should be 9.")


res = a.addAlternate(4, 5)
if res != 9:
    raise RuntimeError, ("SomeStruct::addAlternate(4,5) returns ",
                         res, " should be 9.")
