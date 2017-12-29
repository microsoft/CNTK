import li_std_vector_enum.*;

public class li_std_vector_enum_runme {

  static {
    try {
        System.loadLibrary("li_std_vector_enum");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) throws Throwable
  {
    EnumVector ev = new EnumVector();

    check(ev.getNums().get(0).swigValue(), 10);
    check(ev.getNums().get(1).swigValue(), 20);
    check(ev.getNums().get(2).swigValue(), 30);
  }

  private static void check(int a, int b) {
    if (a != b)
      throw new RuntimeException("values don't match");
  }
}
