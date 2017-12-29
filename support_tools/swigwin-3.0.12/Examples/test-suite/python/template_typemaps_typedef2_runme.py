from template_typemaps_typedef2 import *

m1 = MultimapIntA()

dummy_pair = m1.make_dummy_pair()
val = m1.typemap_test(dummy_pair).val
if val != 1234:
    raise RuntimeError, "typemaps not working"

m2 = MultimapAInt()

# TODO: typemaps and specializations not quite working as expected. T needs expanding, but at least the right typemap is being picked up.
#dummy_pair = m2.make_dummy_pair()
#val = m2.typemap_test(dummy_pair)

# print val
# if val != 4321:
#    raise RuntimeError, "typemaps not working"

if typedef_test1(dummy_pair).val != 1234:
    raise RuntimeError, "typedef_test1 not working"

if typedef_test2(dummy_pair).val != 1234:
    raise RuntimeError, "typedef_test2 not working"

if typedef_test3(dummy_pair).val != 1234:
    raise RuntimeError, "typedef_test3 not working"

if typedef_test4(dummy_pair).val != 1234:
    raise RuntimeError, "typedef_test4 not working"

if typedef_test5(dummy_pair).val != 1234:
    raise RuntimeError, "typedef_test5 not working"

if typedef_test6(dummy_pair).val != 1234:
    raise RuntimeError, "typedef_test6 not working"
