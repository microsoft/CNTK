

import template_template_parameters.*;

public class template_template_parameters_runme {

  static {
    try {
	System.loadLibrary("template_template_parameters");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) {
    ListFastBool listBool = new ListFastBool();
    listBool.setItem(true);
    if (listBool.getItem() != true)
      throw new RuntimeException("Failed");

    ListDefaultDouble listDouble = new ListDefaultDouble();
    listDouble.setItem(10.2);
    if (listDouble.getItem() != 10.2)
      throw new RuntimeException("Failed");
  }
}

