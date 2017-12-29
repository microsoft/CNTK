
import java_director_assumeoverride.*;

public class java_director_assumeoverride_runme {

  static {
    try {
      System.loadLibrary("java_director_assumeoverride");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  static class MyOverrideMe extends OverrideMe {
  }

  public static void main(String argv[]) {
    OverrideMe overrideMe = new MyOverrideMe();

    // MyOverrideMe doesn't actually override func(), but because assumeoverride
    // was set to true, the C++ side will believe it was overridden.
    if (!java_director_assumeoverride.isFuncOverridden(overrideMe)) {
      throw new RuntimeException ( "isFuncOverridden()" );
    }
  }
}
