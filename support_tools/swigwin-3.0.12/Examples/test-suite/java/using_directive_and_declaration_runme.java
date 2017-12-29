
import using_directive_and_declaration.*;

public class using_directive_and_declaration_runme {

  static {
    try {
        System.loadLibrary("using_directive_and_declaration");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) 
  {
    using_directive_and_declaration.useit1(new Thing1());
    using_directive_and_declaration.useit2(new Thing2());
    using_directive_and_declaration.useit3(new Thing3());
    using_directive_and_declaration.useit4(new Thing4());
    using_directive_and_declaration.useit5(new Thing5());
    Thing6a t6a = new Thing6a();
    t6a.a();
    Thing6 t6b = new Thing6();
    t6b.b();
    using_directive_and_declaration.useit6(t6a, t6b);
    using_directive_and_declaration.useit7(new Thing7());
  }
}
