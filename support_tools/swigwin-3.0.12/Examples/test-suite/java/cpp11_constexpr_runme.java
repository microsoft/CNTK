import cpp11_constexpr.*;

public class cpp11_constexpr_runme {

  static {
    try {
        System.loadLibrary("cpp11_constexpr");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  private static void check(int received, int expected) {
    if (expected != received)
      throw new RuntimeException("check failed, expected: " + expected + " received: " + received);
  }

  public static void main(String argv[])
  {
    check(cpp11_constexpr.getAAA(), 10);
    check(cpp11_constexpr.getBBB(), 20);
    check(cpp11_constexpr.CCC(), 30);
    check(cpp11_constexpr.DDD(), 40);

    ConstExpressions ce = new ConstExpressions(0);
    check(ce.JJJ, 100);
    check(ce.KKK, 200);
    check(ce.LLL, 300);
    check(ce.MMM(), 400);
    check(ce.NNN(), 500);
  }
}
