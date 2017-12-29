
import enum_forward.*;

public class enum_forward_runme {

  static {
    try {
        System.loadLibrary("enum_forward");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) 
  {
    try {
      ForwardEnum1 f1 = enum_forward.get_enum1();
      f1 = enum_forward.test_function1(f1);
    } catch (IllegalArgumentException e) {
    }

    try {
      ForwardEnum2 f2 = enum_forward.get_enum2();
      f2 = enum_forward.test_function2(f2);
    } catch (IllegalArgumentException e) {
    }

    ForwardEnum3 f3 = enum_forward.get_enum3();
    f3 = enum_forward.test_function3(f3);
  }
}

