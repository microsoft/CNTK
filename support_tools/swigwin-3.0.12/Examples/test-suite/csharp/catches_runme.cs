using System;
using catchesNamespace;

public class runme {
  static void Main() {
    // test_catches()
    try {
      catches.test_catches(1);
      throw new Exception("missed exception");
    } catch (ApplicationException e) {
      if (e.Message != "C++ int exception thrown, value: 1")
        throw new ApplicationException("bad exception order: " + e.Message);
    }

    try {
      catches.test_catches(2);
      throw new Exception("missed exception");
    } catch (ApplicationException e) {
      if (e.Message != "two")
        throw new ApplicationException("bad exception order: " + e.Message);
    }

    try {
      catches.test_catches(3);
      throw new Exception("missed exception");
    } catch (ApplicationException e) {
      if (e.Message != "C++ ThreeException const & exception thrown")
        throw new ApplicationException("bad exception order: " + e.Message);
    }

    // test_exception_specification()
    try {
      catches.test_exception_specification(1);
      throw new Exception("missed exception");
    } catch (ApplicationException e) {
      if (e.Message != "C++ int exception thrown, value: 1")
        throw new ApplicationException("bad exception order: " + e.Message);
    }

    try {
      catches.test_exception_specification(2);
      throw new Exception("missed exception");
    } catch (ApplicationException e) {
      if (e.Message != "unknown exception")
        throw new ApplicationException("bad exception order: " + e.Message);
    }

    try {
      catches.test_exception_specification(3);
      throw new Exception("missed exception");
    } catch (ApplicationException e) {
      if (e.Message != "unknown exception")
        throw new ApplicationException("bad exception order: " + e.Message);
    }

    // test_catches_all()
    try {
      catches.test_catches_all(1);
      throw new Exception("missed exception");
    } catch (ApplicationException e) {
      if (e.Message != "unknown exception")
        throw new ApplicationException("bad exception order: " + e.Message);
    }

  }
}
