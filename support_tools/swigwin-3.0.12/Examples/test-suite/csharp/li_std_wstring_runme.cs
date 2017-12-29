using System;
using li_std_wstringNamespace;

public class runme
{
    static void Main() 
    {
      char y='h';

      if (li_std_wstring.test_wcvalue(y) != y)
        throw new Exception("bad string mapping:" + li_std_wstring.test_wcvalue(y));

      if (li_std_wstring.test_wcvalue_w() != 'W')
        throw new Exception("bad string mapping:" + li_std_wstring.test_wcvalue_w());

      string x="hello";

      if (li_std_wstring.test_ccvalue(x) != x)
        throw new Exception("bad string mapping");

      if (li_std_wstring.test_cvalue(x) != x)
        throw new Exception("bad string mapping");


      if (li_std_wstring.test_value(x) != x)
        throw new Exception("bad string mapping: " + x + li_std_wstring.test_value(x));

      if (li_std_wstring.test_const_reference(x) != x)
        throw new Exception("bad string mapping");


      string s = "he";
      s = s + "llo";

      if (s != x)
        throw new Exception("bad string mapping: " + s + x);

      if (li_std_wstring.test_value(s) != x)
        throw new Exception("bad string mapping");

      if (li_std_wstring.test_const_reference(s) != x)
        throw new Exception("bad string mapping");

      string a = s;

      if (li_std_wstring.test_value(a) != x)
        throw new Exception("bad string mapping");

      if (li_std_wstring.test_const_reference(a) != x)
        throw new Exception("bad string mapping");

      string b = " world";

      if (a + b != "hello world")
        throw new Exception("bad string mapping");

      if (a + " world" != "hello world")
        throw new Exception("bad string mapping");

      if ("hello" + b != "hello world")
        throw new Exception("bad string mapping");

      s = "hello world";

      B myB = new B("hi");

      myB.name = "hello";
      if (myB.name != "hello")
        throw new Exception("bad string mapping");

      myB.a = "hello";
      if (myB.a != "hello")
        throw new Exception("bad string mapping");
    }
}

