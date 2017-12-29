
public class runme {
  static {
    try {
        System.loadLibrary("example");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) 
  {
    // Print out the value of some enums
    System.out.println("*** color ***");
    System.out.println("    " + color.RED + " = " + color.RED.swigValue());
    System.out.println("    " + color.BLUE + " = " + color.BLUE.swigValue());
    System.out.println("    " + color.GREEN + " = " + color.GREEN.swigValue());

    System.out.println("\n*** Foo::speed ***");
    System.out.println("    Foo::" + Foo.speed.IMPULSE + " = " + Foo.speed.IMPULSE.swigValue());
    System.out.println("    Foo::" + Foo.speed.WARP + " = " + Foo.speed.WARP.swigValue());
    System.out.println("    Foo::" + Foo.speed.LUDICROUS + " = " + Foo.speed.LUDICROUS.swigValue());

    System.out.println("\nTesting use of enums with functions\n");

    example.enum_test(color.RED, Foo.speed.IMPULSE);
    example.enum_test(color.BLUE, Foo.speed.WARP);
    example.enum_test(color.GREEN, Foo.speed.LUDICROUS);

    System.out.println( "\nTesting use of enum with class method" );
    Foo f = new Foo();

    f.enum_test(Foo.speed.IMPULSE);
    f.enum_test(Foo.speed.WARP);
    f.enum_test(Foo.speed.LUDICROUS);
  }
}
