
import namespace_forward_declaration.*;

public class namespace_forward_declaration_runme {

  static {
    try {
	System.loadLibrary("namespace_forward_declaration");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) {
    XXX xxx = new XXX();
    namespace_forward_declaration.testXXX1(xxx);
    namespace_forward_declaration.testXXX2(xxx);
    namespace_forward_declaration.testXXX3(xxx);
    YYY yyy = new YYY();
    namespace_forward_declaration.testYYY1(yyy);
    namespace_forward_declaration.testYYY2(yyy);
    namespace_forward_declaration.testYYY3(yyy);
  }
}

