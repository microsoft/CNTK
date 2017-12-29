import sizet.*;

public class sizet_runme {

  static {
    try {
        System.loadLibrary("sizet");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) throws Throwable
  {
    long s = 2000;
    s = sizet.test1(s+1);
    s = sizet.test2(s+1);
    s = sizet.test3(s+1);
    s = sizet.test4(s+1);
    if (s != 2004)
      throw new RuntimeException("failed");
  }
}
