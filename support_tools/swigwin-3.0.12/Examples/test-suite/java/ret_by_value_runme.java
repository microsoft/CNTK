
// This is the ret_by_value runtime testcase. It checks that SWIG handles
// return by value okay.

import ret_by_value.*;

public class ret_by_value_runme {

  static {
    try {
	System.loadLibrary("ret_by_value");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) {

    // Get the test class. Note that this constructor will ensure that the memory created 
    // in the wrapper is owned by the test class.
    test tst = ret_by_value.get_test();

    if (tst.getMyInt() != 100 || tst.getMyShort() != 200) {
      System.err.println("Runtime test failed. myInt=" + tst.getMyInt() + " myShort=" + tst.getMyShort());
      System.exit(1);
    }

    // Delete memory manually, it should not be deleted again by the test class finalizer
    tst.delete();
  }
}

