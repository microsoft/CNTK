

import template_typedef_inherit.*;

public class template_typedef_inherit_runme {

  static {
    try {
	System.loadLibrary("template_typedef_inherit");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) {
    DescriptionImplementationTypedCollectionInterfaceObject d = new DescriptionImplementationTypedCollectionInterfaceObject();
    d.add("a string");

    StringPersistentCollection s = new StringPersistentCollection();
    s.add("a string");
  }
}

