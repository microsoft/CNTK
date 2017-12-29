import proxycode.*;

public class proxycode_runme {

  static {
    try {
        System.loadLibrary("proxycode");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) throws Throwable
  {
    if (new Proxy1().proxycode1(100) != 101)
      throw new RuntimeException("Fail");
    if (new Proxy2().proxycode2a(100) != 102)
      throw new RuntimeException("Fail");
    if (new Proxy2().proxycode2b(100) != 102)
      throw new RuntimeException("Fail");
    if (new Proxy3().proxycode3(100) != 103)
      throw new RuntimeException("Fail");

    if (new Proxy4().proxycode4(100) != 104)
      throw new RuntimeException("Fail");
    if (new Proxy4.Proxy4Nested().proxycode4nested(100) != 144)
      throw new RuntimeException("Fail");

    if (new Proxy5a().proxycode5((short)100) != (short)100)
      throw new RuntimeException("Fail");
    if (new Proxy5b().proxycode5(100) != 100)
      throw new RuntimeException("Fail");
    if (new Proxy5b().proxycode5(100, 100) != 255)
      throw new RuntimeException("Fail");

    long t1 = 10;
    long t2 = 100;
    Proxy6 p = new Proxy6().proxyUseT(t1, t2);
    p.useT(t1, t2);
  }
}
