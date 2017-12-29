// This test tests all the methods in the C# collection wrapper

using System;
using pointer_referenceNamespace;

public class pointer_reference_runme {

  public static void Main() {
    Struct s  = pointer_reference.get();
    if (s.value != 10) throw new Exception("get test failed");

    Struct ss = new Struct(20);
    pointer_reference.set(ss);
    if (Struct.instance.value != 20) throw new Exception("set test failed");

    if (pointer_reference.overloading(1) != 111) throw new Exception("overload test 1 failed");
    if (pointer_reference.overloading(ss) != 222) throw new Exception("overload test 2 failed");
  }

}

