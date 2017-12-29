from cpp11_template_typedefs import *

t = create_TypedefName()
if type(t).__name__ != "SomeTypeInt5":
    raise RuntimeError("type(t) is '%s' and should be 'SomeTypeInt5'" % type(t).__name__)

if t.a != "hello":
    raise RuntimeError("t.a should be 'hello'")
if t.b != 10:
    raise RuntimeError("t.b should be 10")
if t.get_n() != 5:
    raise RuntimeError("t.get_n() should be 5")

t_bool = create_TypedefNameBool()
if type(t_bool).__name__ != "SomeTypeBool5":
    raise RuntimeError("type(t_bool) is '%s' and should be 'SomeTypeBool5'" % type(t_bool).__name__)

if t_bool.a != "hello":
    raise RuntimeError("t_bool.a should be 'hello'")
if t_bool.b != True:
    raise RuntimeError("t_bool.b should be True")
if t_bool.get_n() != 15:
    raise RuntimeError("t_bool.get_n() should be 15")

if get_SomeType_b(t) != 10:
    raise RuntimeError("get_SomeType_b(t) should be 10")

if get_SomeType_b2(t) != 10:
    raise RuntimeError("get_SomeType_b2(t) should be 10")

t2 = SomeTypeInt4()
t2.b = 0
t3 = identity(t2)
t3.b = 5
if t2.b != 5:
    raise RuntimeError("t2.b should be 5")

if get_bucket_allocator1() != 1:
    raise RuntimeError("bucket_allocator1 should be 1")

# SWIG doesn't handle ::MyClass as a template argument. Skip this test.
#if get_bucket_allocator2() != 2:
#    raise RuntimeError("bucket_allocator2 should be 2")
