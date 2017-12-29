
public class runme {

  static {
    try {
	System.loadLibrary("example");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) {

    // First create some objects using the pointer library.
    System.out.println("Testing the pointer library");
    SWIGTYPE_p_int a = example.new_intp();
    SWIGTYPE_p_int b = example.new_intp();
    SWIGTYPE_p_int c = example.new_intp();
    example.intp_assign(a,37);
    example.intp_assign(b,42);

    // Note that getCPtr() has package access by default
    System.out.println("     a =" + Long.toHexString(SWIGTYPE_p_int.getCPtr(a)));
    System.out.println("     b =" + Long.toHexString(SWIGTYPE_p_int.getCPtr(b)));
    System.out.println("     c =" + Long.toHexString(SWIGTYPE_p_int.getCPtr(c)));

    // Call the add() function with some pointers
    example.add(a,b,c);

    // Now get the result
    int res = example.intp_value(c);
    System.out.println("     37 + 42 =" + res);

    // Clean up the pointers
    example.delete_intp(a);
    example.delete_intp(b);
    example.delete_intp(c);

    // Now try the typemap library
    // Now it is no longer necessary to manufacture pointers.
    // Instead we use a single element array which in Java is modifiable.

    System.out.println("Trying the typemap library");
    int[] r = {0};
    example.sub(37,42,r);
    System.out.println("     37 - 42 = " + r[0]);

    // Now try the version with return value

    System.out.println("Testing return value");
    int q = example.divide(42,37,r);
    System.out.println("     42/37 = " + q + " remainder " + r[0]);
  }
}
