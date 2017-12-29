
import template_namespace_forward_declaration.*;

public class template_namespace_forward_declaration_runme {

  static {
    try {
	System.loadLibrary("template_namespace_forward_declaration");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) {
    XXXInt xxx = new XXXInt();
    template_namespace_forward_declaration.testXXX1(xxx);
    template_namespace_forward_declaration.testXXX2(xxx);
    template_namespace_forward_declaration.testXXX3(xxx);
    YYYInt yyy = new YYYInt();
    template_namespace_forward_declaration.testYYY1(yyy);
    template_namespace_forward_declaration.testYYY2(yyy);
    template_namespace_forward_declaration.testYYY3(yyy);
  }
}

