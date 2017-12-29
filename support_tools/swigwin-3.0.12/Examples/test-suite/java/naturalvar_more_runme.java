
import naturalvar_more.*;

public class naturalvar_more_runme {
  static {
    try {
        System.loadLibrary("naturalvar_more");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) 
  {
    S s = new S();
    if (!s.getConst_string_member().equals("initial string value"))
      throw new RuntimeException("Test 1 fail");
    s.setString_member("some member value");
    if (!s.getString_member().equals("some member value"))
      throw new RuntimeException("Test 2 fail");
  }
}
