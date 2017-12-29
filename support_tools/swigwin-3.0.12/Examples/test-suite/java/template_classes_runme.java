
// This is the template_classes runtime testcase. It checks that SWIG handles a templated 
// class used by another templated class, in particular that the proxy classes can be used.

import template_classes.*;

public class template_classes_runme {

  static {
    try {
	System.loadLibrary("template_classes");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) {

    RectangleInt rectint = new RectangleInt();
    PointInt pi = rectint.getPoint();
    int x = pi.getX();
  }
}

