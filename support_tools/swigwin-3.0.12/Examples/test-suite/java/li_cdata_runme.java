import li_cdata.*;

public class li_cdata_runme {

  static {
    try {
        System.loadLibrary("li_cdata");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) throws Throwable
  {
    byte[] s = "ABC abc".getBytes();
    SWIGTYPE_p_void m = li_cdata.malloc(256);
    li_cdata.memmove(m, s);
    byte[] ss = li_cdata.cdata(m, 7);
    String ss_string = new String(ss);
    if (!ss_string.equals("ABC abc"))
      throw new RuntimeException("failed got: " + ss_string);
  }
}
