import typemap_arrays.*;

public class typemap_arrays_runme {

  static {
    try {
	System.loadLibrary("typemap_arrays");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) {
    if (typemap_arrays.sumA(null) != 60)
      throw new RuntimeException("Sum is wrong");
  }
}

