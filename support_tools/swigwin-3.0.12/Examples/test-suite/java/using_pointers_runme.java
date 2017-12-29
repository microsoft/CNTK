
import using_pointers.*;

public class using_pointers_runme {

  static {
    try {
        System.loadLibrary("using_pointers");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) 
  {
    FooBar f = new FooBar();
    boolean pass = true;
    try {
      f.exception_spec(1);
      pass = false;
    } catch (RuntimeException e) {
    }
    if (!pass) throw new RuntimeException("Missed exception 1");
    try {
      f.exception_spec(2);
      pass = false;
    } catch (RuntimeException e) {
    }
    if (!pass) throw new RuntimeException("Missed exception 2");
  }
}
