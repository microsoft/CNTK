using System;
using exception_orderNamespace;

public class runme {
  static void Main() {
    A a = new A();

    try {
      a.foo();
      throw new Exception("missed exception");
    } catch (ApplicationException e) {
      if (e.Message != "C++ E1 exception thrown")
        throw new ApplicationException("bad exception order: " + e.Message);
    }

    try {
      a.bar();
      throw new Exception("missed exception");
    } catch (ApplicationException e) {
      if (e.Message != "C++ E2 exception thrown")
        throw new ApplicationException("bad exception order: " + e.Message);
    }

    try {
      a.foobar();
      throw new Exception("missed exception");
    } catch (ApplicationException e) {
      if (e.Message != "postcatch unknown")
        throw new ApplicationException("bad exception order: " + e.Message);
    }

    try {
      a.barfoo(1);
      throw new Exception("missed exception");
    } catch (ApplicationException e) {
      if (e.Message != "C++ E1 exception thrown")
        throw new ApplicationException("bad exception order: " + e.Message);
    }

    try {
      a.barfoo(2);
      throw new Exception("missed exception");
    } catch (ApplicationException e) {
      if (e.Message != "C++ E2 * exception thrown")
        throw new ApplicationException("bad exception order: " + e.Message);
    }
  }
}
