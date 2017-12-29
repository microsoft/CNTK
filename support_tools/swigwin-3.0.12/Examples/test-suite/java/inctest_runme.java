
import inctest.*;

public class inctest_runme {
  static {
    try {
        System.loadLibrary("inctest");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) 
  {
    MY_THINGS things = new MY_THINGS();
    int i=0;
    things.setIntegerMember(i);
    double d = things.getDoubleMember();
  }
}
