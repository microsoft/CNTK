using System;
using director_stringNamespace;

public class runme
{
  static void Main() 
  {
    runme r = new runme();
    r.run();
  }

  void run()
  {
    String s;

    director_string_A c = new director_string_A("hi");
    for (int i=0; i<3; i++) {
      s = c.call_get(i);
      Object ii = i;
      if (s != ii.ToString()) throw new Exception("director_string_A.get(" + i + ") failed. Got:" + s);
    }

    director_string_B b = new director_string_B("hello");

    s = b.call_get_first();
    if (s != "director_string_B.get_first") throw new Exception("call_get_first() failed");

    s = b.call_get(0);
    if (s != "director_string_B.get: hello") throw new Exception("get(0) failed");
  }
}

class director_string_B : A {
    public director_string_B(String first) : base(first) {
    }
    public override String get_first() {
      return "director_string_B.get_first";
    }
  
    public override String get(int n) {
      return "director_string_B.get: " + base.get(n);
    }
}

class director_string_A : A {
    public director_string_A(String first) : base(first) {
    }
    public override String get(int n) {
      Object nn = n;
      return nn.ToString();
    }
}

