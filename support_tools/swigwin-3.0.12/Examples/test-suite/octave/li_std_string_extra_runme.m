li_std_string_extra

x="hello";



if (li_std_string_extra.test_ccvalue(x) != x)
  error("bad string mapping")
endif

if (li_std_string_extra.test_cvalue(x) != x)
  error("bad string mapping")
endif

if (li_std_string_extra.test_value(x) != x)
  error("bad string mapping: %s, %s", x, li_std_string_extra.test_value(x))
endif

if (li_std_string_extra.test_const_reference(x) != x)
  error("bad string mapping")
endif


s = li_std_string_extra.string("he");
#s += "ll"
#s.append("ll")
s = s + "llo";

if (s != x)
  error("bad string mapping: %s, %s", s, x);
endif

#if (s(1:4) != x(1:4))
#  error("bad string mapping")
#endif

if (li_std_string_extra.test_value(s) != x)
  error("bad string mapping")
endif

if (li_std_string_extra.test_const_reference(s) != x)
  error("bad string mapping")
endif

a = li_std_string_extra.A(s);

if (li_std_string_extra.test_value(a) != x)
  error("bad string mapping")
endif

if (li_std_string_extra.test_const_reference(a) != x)
  error("bad string mapping")
endif

b = li_std_string_extra.string(" world");

s = a + b;
if (a + b != "hello world")
  error("bad string mapping: %s", a + b)
endif
  
if (a + " world" != "hello world")
  error("bad string mapping")
endif

#if ("hello" + b != "hello world")
#  error("bad string mapping")
#endif

c = (li_std_string_extra.string("hello") + b);
if (c.find_last_of("l") != 9)
  error("bad string mapping")
endif
  
s = "hello world";

b = li_std_string_extra.B("hi");

b.name = li_std_string_extra.string("hello");
if (b.name != "hello")
  error("bad string mapping")
endif


b.a = li_std_string_extra.A("hello");
if (b.a != "hello")
  error("bad string mapping")
endif


if (li_std_string_extra.test_value_basic1(x) != x)
  error("bad string mapping")
endif

if (li_std_string_extra.test_value_basic2(x) != x)
  error("bad string mapping")
endif


if (li_std_string_extra.test_value_basic3(x) != x)
  error("bad string mapping")
endif

# Global variables
s = "initial string";
if (li_std_string_extra.cvar.GlobalString2 != "global string 2")
  error("GlobalString2 test 1")
endif
li_std_string_extra.cvar.GlobalString2 = s;
if (li_std_string_extra.cvar.GlobalString2 != s)
  error("GlobalString2 test 2")
endif
if (li_std_string_extra.cvar.ConstGlobalString != "const global string")
  error("ConstGlobalString test")
endif

# Member variables
myStructure = li_std_string_extra.Structure();
if (myStructure.MemberString2 != "member string 2")
  error("MemberString2 test 1")
endif
myStructure.MemberString2 = s;
if (myStructure.MemberString2 != s)
  error("MemberString2 test 2")
endif
if (myStructure.ConstMemberString != "const member string")
  error("ConstMemberString test")
endif

if (li_std_string_extra.cvar.Structure_StaticMemberString2 != "static member string 2")
  error("StaticMemberString2 test 1")
endif
li_std_string_extra.cvar.Structure_StaticMemberString2 = s;
if (li_std_string_extra.cvar.Structure_StaticMemberString2 != s)
  error("StaticMemberString2 test 2")
endif
if (li_std_string_extra.cvar.Structure_ConstStaticMemberString != "const static member string")
  error("ConstStaticMemberString test")
endif


if (li_std_string_extra.test_reference_input("hello") != "hello")
  error
endif
s = li_std_string_extra.test_reference_inout("hello");
if (s != "hellohello")
  error
endif


if (li_std_string_extra.stdstring_empty() != "")
  error
endif


if (li_std_string_extra.c_empty() != "")
  error
endif

#if (li_std_string_extra.c_null() != None)
#  error
#endif
