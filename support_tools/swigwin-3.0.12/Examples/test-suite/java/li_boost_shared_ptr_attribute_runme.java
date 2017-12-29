import li_boost_shared_ptr_attribute.*;

public class li_boost_shared_ptr_attribute_runme {
  static {
    try {
        System.loadLibrary("li_boost_shared_ptr_attribute");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void check(GetSetMe g, int expected) {
    int got = g.getN();
    if (got != expected)
      throw new RuntimeException("GetSetMe value is " + got + " but should be " + expected);
  }

  public static void check(GetMe g, int expected) {
    int got = g.getN();
    if (got != expected)
      throw new RuntimeException("GetMe value is " + got + " but should be " + expected);
  }

  public static void main(String argv[])
  {
    GetterSetter gs = new GetterSetter(5);
    check(gs.getMyval(), 25);
    check(gs.getAddedAttrib(), 25);
    gs.setAddedAttrib(new GetSetMe(6));
    check(gs.getMyval(), 6);
    check(gs.getAddedAttrib(), 6);

    GetterOnly g = new GetterOnly(4);
    check(g.getMyval(), 16);
    check(g.getAddedAttrib(), 16);
  }
}
