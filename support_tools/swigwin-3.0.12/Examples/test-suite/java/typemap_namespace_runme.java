import typemap_namespace.*;

public class typemap_namespace_runme {

  static {
    try {
        System.loadLibrary("typemap_namespace");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) throws Throwable
  {
    if (!typemap_namespace.test1("hello").equals("hello"))
      throw new RuntimeException("test1 failed");
    if (!typemap_namespace.test2("hello").equals("hello"))
      throw new RuntimeException("test2 failed");
  }
}
