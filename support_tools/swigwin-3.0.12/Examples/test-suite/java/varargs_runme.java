// varargs test

import varargs.*;

public class varargs_runme {

  static {
    try {
	System.loadLibrary("varargs");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) {

    if (!varargs.test("Hello").equals("Hello"))
      throw new RuntimeException("Failed");

    Foo f = new Foo("BuonGiorno", 1);
    if (!f.getStr().equals("BuonGiorno"))
      throw new RuntimeException("Failed");

    f = new Foo("Greetings");
    if (!f.getStr().equals("Greetings"))
      throw new RuntimeException("Failed");
        
    if (!f.test("Hello").equals("Hello"))
      throw new RuntimeException("Failed");

    if (!Foo.statictest("Grussen", 1).equals("Grussen"))
      throw new RuntimeException("Failed");
  }
}
