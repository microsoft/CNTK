import extend_constructor_destructor.*;

public class extend_constructor_destructor_runme {

  static {
    try {
	System.loadLibrary("extend_constructor_destructor");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) {
    AStruct a = new AStruct(10);
    checkGlobal(10);
    BStruct b = new BStruct(20);
    checkGlobal(20);
    CStruct c = new CStruct(30);
    checkGlobal(30);
    DStruct d = new DStruct(40);
    checkGlobal(40);
    EStruct e = new EStruct(50);
    checkGlobal(50);
    FStruct f = new FStruct(60);
    checkGlobal(60);
    GStruct g = new GStruct(70);
    checkGlobal(70);

    a.delete();
    checkGlobal(-10);
    b.delete();
    checkGlobal(-20);
    c.delete();
    checkGlobal(-30);
    d.delete();
    checkGlobal(-40);
    e.delete();
    checkGlobal(-50);
    f.delete();
    checkGlobal(-60);
    g.delete();
    checkGlobal(-70);
  }

  public static void checkGlobal(int val) {
    int global = extend_constructor_destructor.getGlobalVar();
    if (global != val)
      throw new RuntimeException("global value incorrect. Expected: " + val + " got: " + global);
  }
}

