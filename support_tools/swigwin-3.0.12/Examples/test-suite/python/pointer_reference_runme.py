import pointer_reference

s = pointer_reference.get()
if s.value != 10:
    raise RuntimeError, "get test failed"

ss = pointer_reference.Struct(20)
pointer_reference.set(ss)
if pointer_reference.cvar.Struct_instance.value != 20:
    raise RuntimeError, "set test failed"

if pointer_reference.overloading(1) != 111:
    raise RuntimeError, "overload test 1 failed"

if pointer_reference.overloading(ss) != 222:
    raise RuntimeError, "overload test 2 failed"
