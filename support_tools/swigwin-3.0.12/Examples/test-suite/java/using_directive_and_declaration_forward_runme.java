
import using_directive_and_declaration_forward.*;

public class using_directive_and_declaration_forward_runme {

  static {
    try {
        System.loadLibrary("using_directive_and_declaration_forward");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) 
  {
    using_directive_and_declaration_forward.useit1(new Thing1());
    using_directive_and_declaration_forward.useit1a(new Thing1());
    using_directive_and_declaration_forward.useit1b(new Thing1());
    using_directive_and_declaration_forward.useit1c(new Thing1());

    using_directive_and_declaration_forward.useit2(new Thing2());
    using_directive_and_declaration_forward.useit2a(new Thing2());
    using_directive_and_declaration_forward.useit2b(new Thing2());
    using_directive_and_declaration_forward.useit2c(new Thing2());
    using_directive_and_declaration_forward.useit2d(new Thing2());

    using_directive_and_declaration_forward.useit3(new Thing3());
    using_directive_and_declaration_forward.useit3a(new Thing3());
    using_directive_and_declaration_forward.useit3b(new Thing3());
    using_directive_and_declaration_forward.useit3c(new Thing3());
    using_directive_and_declaration_forward.useit3d(new Thing3());

    using_directive_and_declaration_forward.useit4(new Thing4());
    using_directive_and_declaration_forward.useit4a(new Thing4());
    using_directive_and_declaration_forward.useit4b(new Thing4());
    using_directive_and_declaration_forward.useit4c(new Thing4());
    using_directive_and_declaration_forward.useit4d(new Thing4());

    using_directive_and_declaration_forward.useit5(new Thing5());
    using_directive_and_declaration_forward.useit5a(new Thing5());
    using_directive_and_declaration_forward.useit5b(new Thing5());
    using_directive_and_declaration_forward.useit5c(new Thing5());
    using_directive_and_declaration_forward.useit5d(new Thing5());


    using_directive_and_declaration_forward.useit7(new Thing7());
    using_directive_and_declaration_forward.useit7a(new Thing7());
    using_directive_and_declaration_forward.useit7b(new Thing7());
    using_directive_and_declaration_forward.useit7c(new Thing7());
    using_directive_and_declaration_forward.useit7d(new Thing7());
  }
}
