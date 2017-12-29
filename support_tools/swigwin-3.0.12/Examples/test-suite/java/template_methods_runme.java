
import template_methods.*;

public class template_methods_runme {

  static {
    try {
	System.loadLibrary("template_methods");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) {
    float num = (float)1.1;

    // Global templated functions
    int i = template_methods.convolve1Bool();
    template_methods.convolve1Bool(true);
    i = template_methods.convolve2Float();
    template_methods.convolve3FloatRenamed(num);
    i = template_methods.convolve4Float();
    template_methods.convolve4FloatRenamed(num);
    i = template_methods.convolve5FloatRenamed();
    template_methods.convolve5FloatRenamed(num);


    // Static templated methods
    Klass k = new Klass();
    boolean b = k.KlassTMethodBoolRenamed(true);
    k.KlassTMethodBool();
    b = Klass.KlassStaticTMethodBoolRenamed(true);
    Klass.KlassStaticTMethodBool();


    //
    ComponentProperties cp = new ComponentProperties();
    cp.adda("key1", "val1", "key2", 22.2);
    cp.adda("key1", "val1", "key2", "val2", "key3", "val3");
    cp.adda("key1", 1, "key2", 2, "key3", 3);
  }
}

