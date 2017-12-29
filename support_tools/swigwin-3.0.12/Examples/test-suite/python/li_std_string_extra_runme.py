import li_std_string_extra

x = "hello"


if li_std_string_extra.test_ccvalue(x) != x:
    raise RuntimeError, "bad string mapping"

if li_std_string_extra.test_cvalue(x) != x:
    raise RuntimeError, "bad string mapping"

if li_std_string_extra.test_value(x) != x:
    print x, li_std_string_extra.test_value(x)
    raise RuntimeError, "bad string mapping"

if li_std_string_extra.test_const_reference(x) != x:
    raise RuntimeError, "bad string mapping"


s = li_std_string_extra.string("he")
#s += "ll"
# s.append('o')
s = s + "llo"

if s != x:
    print s, x
    raise RuntimeError, "bad string mapping"

if s[1:4] != x[1:4]:
    raise RuntimeError, "bad string mapping"

if li_std_string_extra.test_value(s) != x:
    raise RuntimeError, "bad string mapping"

if li_std_string_extra.test_const_reference(s) != x:
    raise RuntimeError, "bad string mapping"

a = li_std_string_extra.A(s)

if li_std_string_extra.test_value(a) != x:
    raise RuntimeError, "bad string mapping"

if li_std_string_extra.test_const_reference(a) != x:
    raise RuntimeError, "bad string mapping"

b = li_std_string_extra.string(" world")

s = a + b
if a + b != "hello world":
    print a + b
    raise RuntimeError, "bad string mapping"

if a + " world" != "hello world":
    raise RuntimeError, "bad string mapping"

# This is expected to fail with -builtin option
# Reverse operators not supported in builtin types
if not li_std_string_extra.is_python_builtin():
    if "hello" + b != "hello world":
        raise RuntimeError, "bad string mapping"

    c = "hello" + b
    if c.find_last_of("l") != 9:
        raise RuntimeError, "bad string mapping"

s = "hello world"

b = li_std_string_extra.B("hi")

b.name = li_std_string_extra.string("hello")
if b.name != "hello":
    raise RuntimeError, "bad string mapping"


b.a = li_std_string_extra.A("hello")
if b.a != "hello":
    raise RuntimeError, "bad string mapping"


if li_std_string_extra.test_value_basic1(x) != x:
    raise RuntimeError, "bad string mapping"

if li_std_string_extra.test_value_basic2(x) != x:
    raise RuntimeError, "bad string mapping"


if li_std_string_extra.test_value_basic3(x) != x:
    raise RuntimeError, "bad string mapping"

# Global variables
s = "initial string"
if li_std_string_extra.cvar.GlobalString2 != "global string 2":
    raise RuntimeError, "GlobalString2 test 1"
li_std_string_extra.cvar.GlobalString2 = s
if li_std_string_extra.cvar.GlobalString2 != s:
    raise RuntimeError, "GlobalString2 test 2"
if li_std_string_extra.cvar.ConstGlobalString != "const global string":
    raise RuntimeError, "ConstGlobalString test"

# Member variables
myStructure = li_std_string_extra.Structure()
if myStructure.MemberString2 != "member string 2":
    raise RuntimeError, "MemberString2 test 1"
myStructure.MemberString2 = s
if myStructure.MemberString2 != s:
    raise RuntimeError, "MemberString2 test 2"
if myStructure.ConstMemberString != "const member string":
    raise RuntimeError, "ConstMemberString test"

if li_std_string_extra.cvar.Structure_StaticMemberString2 != "static member string 2":
    raise RuntimeError, "StaticMemberString2 test 1"
li_std_string_extra.cvar.Structure_StaticMemberString2 = s
if li_std_string_extra.cvar.Structure_StaticMemberString2 != s:
    raise RuntimeError, "StaticMemberString2 test 2"
if li_std_string_extra.cvar.Structure_ConstStaticMemberString != "const static member string":
    raise RuntimeError, "ConstStaticMemberString test"


if li_std_string_extra.test_reference_input("hello") != "hello":
    raise RuntimeError
s = li_std_string_extra.test_reference_inout("hello")
if s != "hellohello":
    raise RuntimeError


if li_std_string_extra.stdstring_empty() != "":
    raise RuntimeError


if li_std_string_extra.c_empty() != "":
    raise RuntimeError

if li_std_string_extra.c_null() != None:
    raise RuntimeError
