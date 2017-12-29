import template_nested_typemaps.*;

public class template_nested_typemaps_runme {

  static {
    try {
	System.loadLibrary("template_nested_typemaps");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) {
    BreezeString b = new BreezeString();
    {
      int v = 88;
      short vTypemap = -99;
      if (b.methodInt1(v) != v) throw new RuntimeException("failed");
      if (b.methodInt2(v) != vTypemap) throw new RuntimeException("failed");

      if (template_nested_typemaps.globalInt1(v) != v) throw new RuntimeException("failed");
      if (template_nested_typemaps.globalInt2(v) != v) throw new RuntimeException("failed");
      if (template_nested_typemaps.globalInt3(v) != vTypemap) throw new RuntimeException("failed");
    }

    {
      short v = 88;
      short vTypemap = -77;
      if (b.methodShort1(v) != v) throw new RuntimeException("failed");
      if (b.methodShort2(v) != vTypemap) throw new RuntimeException("failed");

      if (template_nested_typemaps.globalShort1(v) != v) throw new RuntimeException("failed");
      if (template_nested_typemaps.globalShort2(v) != v) throw new RuntimeException("failed");
      if (template_nested_typemaps.globalShort3(v) != vTypemap) throw new RuntimeException("failed");
    }
  }
}

