
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


    int a = 37;
    int b = 42;

    // Now call our C function with a bunch of callbacks

    System.out.println( "Trying some C callback functions" );
    System.out.println( "    a        = " + a );
    System.out.println( "    b        = " + b );
    System.out.println( "    ADD(a,b) = " + example.do_op(a,b,example.ADD) );
    System.out.println( "    SUB(a,b) = " + example.do_op(a,b,example.SUB) );
    System.out.println( "    MUL(a,b) = " + example.do_op(a,b,example.MUL) );

    System.out.println( "Here is what the C callback function classes are called in Java" );
    System.out.println( "    ADD      = " + example.ADD.getClass().getName() );
    System.out.println( "    SUB      = " + example.SUB.getClass().getName() );
    System.out.println( "    MUL      = " + example.MUL.getClass().getName() );
  }
}
