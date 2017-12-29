
import director_pass_by_value.*;

public class director_pass_by_value_runme {

  static {
    try {
      System.loadLibrary("director_pass_by_value");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  private static void WaitForGC() {
    System.gc();
    System.runFinalization();
    try {
      java.lang.Thread.sleep(10);
    } catch (java.lang.InterruptedException e) {
    }
  }

  public static void main(String argv[]) {
    Caller caller = new Caller();
    caller.call_virtualMethod(new director_pass_by_value_Derived());
    {
      int countdown = 5;
      while (true) {
        WaitForGC();
        if (--countdown == 0)
          break;
      };
    }
    // bug was the passByVal 'global' object was destroyed after the call to virtualMethod had finished.
    int ret = director_pass_by_value_runme.passByVal.getVal();
    if (ret != 0x12345678)
      throw new RuntimeException("Bad return value, got " + Integer.toHexString(ret));
  }

  static PassedByValue passByVal;
}

class director_pass_by_value_Derived extends DirectorPassByValueAbstractBase {
  public void virtualMethod(PassedByValue pbv) {
    director_pass_by_value_runme.passByVal = pbv;
  }
}
