import char_binary.*;

public class char_binary_runme {

  static {
    try {
	System.loadLibrary("char_binary");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) {
    Test t = new Test();
    byte[] hile = "hile".getBytes();
    byte[] hil0 = "hil\0".getBytes();
    if (t.strlen(hile) != 4)
      throw new RuntimeException("bad multi-arg typemap");

    if (t.strlen(hil0) != 4)
      throw new RuntimeException("bad multi-arg typemap");

    if (t.ustrlen(hile) != 4)
      throw new RuntimeException("bad multi-arg typemap");

    if (t.ustrlen(hil0) != 4)
      throw new RuntimeException("bad multi-arg typemap");
  }
}
