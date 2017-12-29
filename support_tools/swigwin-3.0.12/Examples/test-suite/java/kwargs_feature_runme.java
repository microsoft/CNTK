import kwargs_feature.*;

public class kwargs_feature_runme {

  static {
    try {
	System.loadLibrary("kwargs_feature");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) {
    // Check normal overloading still works (no compactdefaultargs) if the kwargs feature is used,
    // as the kwargs feature is not supported
    Foo f = new Foo(99);
    if (f.foo() != 1)
      throw new RuntimeException("It went wrong");
    if (Foo.statfoo(2) != 2)
      throw new RuntimeException("It went wrong");
  }
}
