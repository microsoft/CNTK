
import dynamic_cast.*;

public class dynamic_cast_runme {
  static {
    try {
        System.loadLibrary("dynamic_cast");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) 
  {
    Foo f = new Foo();
    Bar b = new Bar();

    Foo x = f.blah();
    Foo y = b.blah();

    // Note it is possible to downcast y with a Java cast.
    String a = dynamic_cast.do_test((Bar)y);
    if (!a.equals("Bar::test")) {
        throw new RuntimeException("Failed!");
    }
  }
}
