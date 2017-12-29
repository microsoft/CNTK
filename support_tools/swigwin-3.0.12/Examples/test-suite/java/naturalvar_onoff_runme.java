
import naturalvar_onoff.*;

public class naturalvar_onoff_runme {
  static {
    try {
        System.loadLibrary("naturalvar_onoff");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[])
  {
    boolean fail = true;
    Vars vars = new Vars();

    fail = true; try {
      vars.setMember1On(null);
    } catch(NullPointerException e) {fail = false;} if (fail) throw new RuntimeException("Failed");

    vars.setMember2Off(null);

    vars.setMember3Off(null);

    fail = true; try {
      vars.setMember3On(null);
    } catch(NullPointerException e) {fail = false;} if (fail) throw new RuntimeException("Failed");

    vars.setMember4Off(null);

    fail = true; try {
      vars.setMember4On(null);
    } catch(NullPointerException e) {fail = false;} if (fail) throw new RuntimeException("Failed");
  }
}
