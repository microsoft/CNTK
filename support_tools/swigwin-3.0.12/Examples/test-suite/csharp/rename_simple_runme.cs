
using System;
using rename_simpleNamespace;

public class rename_simple_runme {

  public static void Main() {
    NewStruct s = new NewStruct();
    check(111, s.NewInstanceVariable, "NewInstanceVariable");
    check(222, s.NewInstanceMethod(), "NewInstanceMethod");
    check(333, NewStruct.NewStaticMethod(), "NewStaticMethod");
    check(444, NewStruct.NewStaticVariable, "NewStaticVariable");
    check(555, rename_simple.NewFunction(), "NewFunction");
    check(666, rename_simple.NewGlobalVariable, "NewGlobalVariable");

    s.NewInstanceVariable = 1111;
    NewStruct.NewStaticVariable = 4444;
    rename_simple.NewGlobalVariable = 6666;

    check(1111, s.NewInstanceVariable, "NewInstanceVariable");
    check(4444, NewStruct.NewStaticVariable, "NewStaticVariable");
    check(6666, rename_simple.NewGlobalVariable, "NewGlobalVariable");
  }

  public static void check(int expected, int actual, string msg) {
    if (expected != actual)
      throw new Exception("Failed: Expected: " + expected + " actual: " + actual + " " + msg);
  }
}

