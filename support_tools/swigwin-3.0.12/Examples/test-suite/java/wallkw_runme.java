
import wallkw.*;

public class wallkw_runme {

  static {
    try {
	System.loadLibrary("wallkw");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) {
    if (!wallkw.c_clone().equals("clone"))
      throw new RuntimeException("clone_c keyword fail");
    if (!wallkw._delegate().equals("delegate"))
      throw new RuntimeException("delegate keyword fail");
    if (!wallkw._pass().equals("pass"))
      throw new RuntimeException("pass keyword fail");
    if (!wallkw._alias().equals("alias"))
      throw new RuntimeException("alias keyword fail");
    if (!wallkw.C_rescue().equals("rescue"))
      throw new RuntimeException("rescue keyword fail");
  }
}
