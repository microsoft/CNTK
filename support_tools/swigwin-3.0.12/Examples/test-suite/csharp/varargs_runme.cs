// varargs test

using System;
using varargsNamespace;

public class varargs_runme {

  public static void Main() {

    if (varargs.test("Hello") != "Hello")
      throw new Exception("Failed");

    Foo f = new Foo("Greetings");
    if (f.str != "Greetings")
      throw new Exception("Failed");
        
    if (f.test("Hello") != "Hello")
      throw new Exception("Failed");
  }
}
