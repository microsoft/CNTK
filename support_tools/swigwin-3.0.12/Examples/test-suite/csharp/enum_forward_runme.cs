using System;
using enum_forwardNamespace;

public class runme {
  static void Main() {
    ForwardEnum1 f1 = enum_forward.get_enum1();
    f1 = enum_forward.test_function1(f1);

    ForwardEnum2 f2 = enum_forward.get_enum2();
    f2 = enum_forward.test_function2(f2);

    ForwardEnum3 f3 = enum_forward.get_enum3();
    f3 = enum_forward.test_function3(f3);
  }
}

