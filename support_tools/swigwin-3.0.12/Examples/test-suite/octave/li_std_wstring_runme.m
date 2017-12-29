# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

li_std_wstring

x="h";

if (li_std_wstring.test_wcvalue(x) != x)
  error("bad string mapping")
endif

x="hello";
if (li_std_wstring.test_ccvalue(x) != x)
  error("bad string mapping")
endif

if (li_std_wstring.test_cvalue(x) != x)
  error("bad string mapping")
endif

if (li_std_wstring.test_value(x) != x)
  error("bad string mapping")
endif

if (li_std_wstring.test_const_reference(x) != x)
  error("bad string mapping")
endif


s = li_std_wstring.wstring("he");
s = s + "llo";

if (s != x)
  error("bad string mapping")
endif

if (s(1:4) != x(1:4))
  error("bad string mapping")
endif

if (li_std_wstring.test_value(s) != x)
  error("bad string mapping")
endif

if (li_std_wstring.test_const_reference(s) != x)
  error("bad string mapping")
endif

a = li_std_wstring.A(s);

if (li_std_wstring.test_value(a) != x)
  error("bad string mapping")
endif

if (li_std_wstring.test_const_reference(a) != x)
  error("bad string mapping")
endif

b = li_std_wstring.wstring(" world");

if (a + b != "hello world")
  error("bad string mapping")
endif
  
if (a + " world" != "hello world")
  error("bad string mapping")
endif

if ("hello" + b != "hello world")
  error("bad string mapping")
endif

c = "hello" + b;
if (c.find_last_of("l") != 9)
  error("bad string mapping")
endif
  
s = "hello world";

b = li_std_wstring.B("hi");

b.name = li_std_wstring.wstring("hello");
if (b.name != "hello")
  error("bad string mapping")
endif


b.a = li_std_wstring.A("hello");
if (b.a != "hello")
  error("bad string mapping")
endif


