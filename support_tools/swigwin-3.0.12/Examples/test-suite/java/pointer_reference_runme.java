import pointer_reference.*;

public class pointer_reference_runme {

  static {
    try {
        System.loadLibrary("pointer_reference");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) throws Throwable
  {
    Struct s  = pointer_reference.get();
    if (s.getValue() != 10) throw new RuntimeException("get test failed");

    Struct ss = new Struct(20);
    pointer_reference.set(ss);
    if (Struct.getInstance().getValue() != 20) throw new RuntimeException("set test failed");

    if (pointer_reference.overloading(1) != 111) throw new RuntimeException("overload test 1 failed");
    if (pointer_reference.overloading(ss) != 222) throw new RuntimeException("overload test 2 failed");
  }
}
