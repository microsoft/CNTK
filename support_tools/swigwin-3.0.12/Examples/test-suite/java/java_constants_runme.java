
import java_constants.*;

public class java_constants_runme {
  static {
    try {
        System.loadLibrary("java_constants");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) 
  {
      int number = 200;

      // Switch statement will only compile if these constants are initialised 
      // from a constant Java value, that is not from a function call
      switch(number) {
          case java_constants.CHINA:
              break;
          case java_constants.BRISTOLS:
              break;
          default:
              break;
      }
  }
}
