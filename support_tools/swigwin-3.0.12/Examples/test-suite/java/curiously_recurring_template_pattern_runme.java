
import curiously_recurring_template_pattern.*;

public class curiously_recurring_template_pattern_runme {

  static {
    try {
	System.loadLibrary("curiously_recurring_template_pattern");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) {
    Derived d = new Derived();
    d.setBase1Param(1).setDerived1Param(10).setDerived2Param(20).setBase2Param(2);

    if (d.getBase1Param() != 1)
      throw new RuntimeException("fail");
    if (d.getDerived1Param() != 10)
      throw new RuntimeException("fail");
    if (d.getBase2Param() != 2)
      throw new RuntimeException("fail");
    if (d.getDerived2Param() != 20)
      throw new RuntimeException("fail");
  }
}

