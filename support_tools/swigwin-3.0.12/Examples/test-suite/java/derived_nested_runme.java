
import derived_nested.*;

public class derived_nested_runme {

  static {
    try {
	System.loadLibrary("derived_nested");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) {
    BB outer = new BB();
    BB.DD d = new BB.DD();
    BB.EE e = new BB.EE();
    outer.getFf_instance().setZ(outer.getFf_instance().getX());
    outer.useEE(e);
  }
}