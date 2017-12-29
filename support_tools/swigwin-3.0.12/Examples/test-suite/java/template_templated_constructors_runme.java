
import template_templated_constructors.*;

public class template_templated_constructors_runme {

  static {
    try {
	System.loadLibrary("template_templated_constructors");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) {
    TConstructor1 t1 = new TConstructor1(123);
    TConstructor2 t2a = new TConstructor2();
    TConstructor2 t2b = new TConstructor2(123);

    TClass1Int tc1 = new TClass1Int(123.4);
    TClass2Int tc2a = new TClass2Int();
    TClass2Int tc2b = new TClass2Int(123.4);

  }
}

