import nested_template_base.*;

public class nested_template_base_runme {

  static {
    try {
	System.loadLibrary("nested_template_base");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) {
    OuterC.InnerS ois = new OuterC.InnerS(123);
    OuterC.InnerC oic = new OuterC.InnerC();

    // Check base method is available
    if (oic.outer(ois).getVal() != 123)
      throw new RuntimeException("Wrong value calling outer");

    // Check non-derived class using base class
    if (oic.innerc().outer(ois).getVal() != 123)
      throw new RuntimeException("Wrong value calling innerc");

  }
}
