import cpp11_null_pointer_constant

a = cpp11_null_pointer_constant.A()

if a._myA != None:
    raise RuntimeError, (
        "cpp11_null_pointer_constant: _myA should be None, but is ", a._myA)

b = cpp11_null_pointer_constant.A()
if a._myA != b._myA:
    raise RuntimeError, (
        "cpp11_null_pointer_constant: a._myA should be the same as b._myA, but ", a._myA, "!=", b._myA)

a._myA = cpp11_null_pointer_constant.A()
if a._myA == None:
    raise RuntimeError, (
        "cpp11_null_pointer_constant: _myA should be object, but is None")
