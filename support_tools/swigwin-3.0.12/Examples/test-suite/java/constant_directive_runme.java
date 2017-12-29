import constant_directive.*;

public class constant_directive_runme {

  static {
    try {
	System.loadLibrary("constant_directive");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) {
    if (constant_directive.TYPE1_CONSTANT1.getVal() != 1)
      throw new RuntimeException("fail");
    if (constant_directive.TYPE1_CONSTANT2.getVal() != 2)
      throw new RuntimeException("fail");
    if (constant_directive.TYPE1_CONSTANT3.getVal() != 3)
      throw new RuntimeException("fail");
    if (constant_directive.TYPE1CONST_CONSTANT1.getVal() != 1)
      throw new RuntimeException("fail");
    if (constant_directive.TYPE1CPTR_CONSTANT1.getVal() != 1)
      throw new RuntimeException("fail");
  }
}
