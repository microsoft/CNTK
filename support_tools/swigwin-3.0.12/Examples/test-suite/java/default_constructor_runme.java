
import default_constructor.*;

public class default_constructor_runme {
  static {
    try {
        System.loadLibrary("default_constructor");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) 
  {
      // calling protected destructor test
      try {
          G g = new G();
          g.delete();
          throw new RuntimeException("Protected destructor exception should have been thrown");
      } catch (UnsupportedOperationException e) {
      }
  }
}
