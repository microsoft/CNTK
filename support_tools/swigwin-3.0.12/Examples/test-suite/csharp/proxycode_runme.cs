using System;
using proxycodeNamespace;

public class proxycode_runme {

  public static void Main() {
    if (new Proxy1().proxycode1(100) != 101)
      throw new Exception("Fail");
    if (new Proxy2().proxycode2a(100) != 102)
      throw new Exception("Fail");
    if (new Proxy2().proxycode2b(100) != 102)
      throw new Exception("Fail");
    if (new Proxy3().proxycode3(100) != 103)
      throw new Exception("Fail");

    if (new Proxy4().proxycode4(100) != 104)
      throw new Exception("Fail");
    if (new Proxy4.Proxy4Nested().proxycode4nested(100) != 144)
      throw new Exception("Fail");

    if (new Proxy5a().proxycode5((short)100) != (short)100)
      throw new Exception("Fail");
    if (new Proxy5b().proxycode5(100) != 100)
      throw new Exception("Fail");
    if (new Proxy5b().proxycode5(100, 100) != 255)
      throw new Exception("Fail");

    uint t1 = 10;
    uint t2 = 100;
    Proxy6 p = new Proxy6().proxyUseT(t1, t2);
    p.useT(t1, t2);
  }
}
