import nested_workaround.*;

public class nested_workaround_runme {

  static {
    try {
	System.loadLibrary("nested_workaround");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) {
    {
      Inner inner = new Inner(5);
      Outer outer = new Outer();
      Inner newInner = outer.doubleInnerValue(inner);
      if (newInner.getValue() != 10)
        throw new RuntimeException("inner failed");
    }

    {
      Outer outer = new Outer();
      Inner inner = outer.createInner(3);
      Inner newInner = outer.doubleInnerValue(inner);
      if (outer.getInnerValue(newInner) != 6)
        throw new RuntimeException("inner failed");
    }
  }
}
