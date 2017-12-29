import cpp11_type_aliasing.*;

public class cpp11_type_aliasing_runme {

  static {
    try {
        System.loadLibrary("cpp11_type_aliasing");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) {
    Halide_Target ht = new GeneratorBase().getTarget();
    Target x = ht.getValue();
    if (x.getBits() != 32)
      throw new RuntimeException("Incorrect bits");
  }
}
