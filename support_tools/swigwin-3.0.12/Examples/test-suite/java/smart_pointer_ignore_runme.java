import smart_pointer_ignore.*;

public class smart_pointer_ignore_runme {

  static {
    try {
      System.loadLibrary("smart_pointer_ignore");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) {
    DerivedPtr d = smart_pointer_ignore.makeDerived();
    d.baseMethod();
    d.derivedMethod();
  }
}
