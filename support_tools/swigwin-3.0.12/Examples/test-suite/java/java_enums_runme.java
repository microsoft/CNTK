
import java_enums.*;

public class java_enums_runme implements stuff {
  static {
    try {
        System.loadLibrary("java_enums");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) 
  {
      int number = 200;

      // Switch statement will only compile if these enums are initialised 
      // from a constant Java value, that is not from a function call
      switch(number) {
          case stuff.FIDDLE:
              break;
          case stuff.STICKS:
              break;
          case stuff.BONGO:
              break;
          case stuff.DRUMS:
              break;
          default:
              break;
      }
      if (stuff.DRUMS != 15)
          throw new RuntimeException("Incorrect value for DRUMS");

      // check typemaps use short for this enum
      short poppycock = nonsense.POPPYCOCK;
      short tst1 = java_enums.test1(poppycock);
      short tst2 = java_enums.test2(poppycock);

      // Check that stuff is an interface and not a class - we can drop the stuff keyword as this class implements the stuff interface
      switch(number) {
          case FIDDLE:
              break;
          case STICKS:
              break;
          case BONGO:
              break;
          case DRUMS:
              break;
          default:
              break;
      }
  }
}
