// This test tests all the methods in the C# collection wrapper

using System;
using typemap_namespaceNamespace;

public class typemap_namespace_runme {

  public static void Main() {
    if (typemap_namespace.test1("hello") != "hello")
      throw new Exception("test1 failed");
    if (typemap_namespace.test2("hello") != "hello")
      throw new Exception("test2 failed");
  }

}

