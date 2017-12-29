import enum_template
if enum_template.MakeETest() != 1:
    raise RuntimeError

if enum_template.TakeETest(0) != None:
    raise RuntimeError
