
import java_prepost.*;

public class java_prepost_runme {

  static {
    try {
	System.loadLibrary("java_prepost");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) 
  {
    // ensure checked exception is generated
    try {
      PrePostThrows ppt = new PrePostThrows(null, true);
    } catch (InstantiationException e) {
    }
  }

  private static void Assert(double d1, double d2) {
    if (d1 != d2)
      throw new RuntimeException("assertion failure. " + d1 + " != " + d2);
  }
}
