using System;
using nested_workaroundNamespace;
#pragma warning disable 219

public class runme {
  static void Main() {
    {
      Inner inner = new Inner(5);
      Outer outer = new Outer();
      Inner newInner = outer.doubleInnerValue(inner);
      if (newInner.getValue() != 10)
        throw new Exception("inner failed");
    }

    {
      Outer outer = new Outer();
      Inner inner = outer.createInner(3);
      Inner newInner = outer.doubleInnerValue(inner);
      if (outer.getInnerValue(newInner) != 6)
        throw new Exception("inner failed");
    }
  }
}
