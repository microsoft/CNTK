// This test tests all the methods in the C# collection wrapper

using System;
using li_std_vector_enumNamespace;

public class li_std_vector_enum_runme {

  public static void Main() {
    EnumVector ev = new EnumVector();

    check((int)ev.nums[0], 10);
    check((int)ev.nums[1], 20);
    check((int)ev.nums[2], 30);

    int expected = 10;
    foreach (EnumVector.numbers val in ev.nums) {
      check((int)val, expected);
      expected += 10;
    }
  }

  private static void check(int a, int b) {
    if (a != b)
      throw new ApplicationException("values don't match");
  }
}

