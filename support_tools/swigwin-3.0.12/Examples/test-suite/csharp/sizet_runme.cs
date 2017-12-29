
using System;
using sizetNamespace;

public class sizet_runme {

  public static void Main() {
    uint s = 2000;
    s = sizet.test1(s+1);
    s = sizet.test2(s+1);
    s = sizet.test3(s+1);
    s = sizet.test4(s+1);
    if (s != 2004)
      throw new Exception("failed");
  }

}

