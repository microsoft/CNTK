import apply_signed_char.*;

public class apply_signed_char_runme {

  static {
    try {
	System.loadLibrary("apply_signed_char");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) {
    byte smallnum = -127;
    if (apply_signed_char.CharValFunction(smallnum) != smallnum) throw new RuntimeException("failed");
    if (apply_signed_char.CCharValFunction(smallnum) != smallnum) throw new RuntimeException("failed");
    if (apply_signed_char.CCharRefFunction(smallnum) != smallnum) throw new RuntimeException("failed");

    apply_signed_char.setGlobalchar(smallnum);
    if (apply_signed_char.getGlobalchar() != smallnum) throw new RuntimeException("failed");
    if (apply_signed_char.getGlobalconstchar() != -110) throw new RuntimeException("failed");

    DirectorTest d = new DirectorTest();
    if (d.CharValFunction(smallnum) != smallnum) throw new RuntimeException("failed");
    if (d.CCharValFunction(smallnum) != smallnum) throw new RuntimeException("failed");
    if (d.CCharRefFunction(smallnum) != smallnum) throw new RuntimeException("failed");

    d.setMemberchar(smallnum);
    if (d.getMemberchar() != smallnum) throw new RuntimeException("failed");
    if (d.getMemberconstchar() != -112) throw new RuntimeException("failed");

  }
}


