import extend_typedef_class.*;

public class extend_typedef_class_runme {

  static {
    try {
	System.loadLibrary("extend_typedef_class");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) {
    // No namespace
    {
      AClass s = new AClass();
      s.setMembervar(10);
      checkMatch(s.getvar(), 10);
    }
    {
      BClass s = new BClass();
      s.setMembervar(20);
      checkMatch(s.getvar(), 20);
    }
    {
      CClass s = new CClass();
      s.setMembervar(30);
      checkMatch(s.getvar(), 30);
    }
    {
      DClass s = new DClass();
      s.setMembervar(40);
      checkMatch(s.getvar(), 40);
    }

    // In namespace
    {
      AStruct s = new AStruct();
      s.setMembervar(10);
      checkMatch(s.getvar(), 10);
    }
    {
      BStruct s = new BStruct();
      s.setMembervar(20);
      checkMatch(s.getvar(), 20);
    }
    {
      CStruct s = new CStruct();
      s.setMembervar(30);
      checkMatch(s.getvar(), 30);
    }
    {
      DStruct s = new DStruct();
      s.setMembervar(40);
      checkMatch(s.getvar(), 40);
    }
  }

  public static void checkMatch(int expected, int got) {
    if (expected != got)
      throw new RuntimeException("Value incorrect. Expected: " + expected + " got: " + got);
  }
}

