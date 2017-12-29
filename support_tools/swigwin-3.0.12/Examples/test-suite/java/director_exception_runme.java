
import director_exception.*;

public class director_exception_runme {

  static {
    try {
      System.loadLibrary("director_exception");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) {

      director_exception_MyFoo a = new director_exception_MyFoo();
      Foo b = director_exception.launder(a);

      try {
          a.pong();
          throw new RuntimeException ( "Failed to catch exception" );
      }
      catch (UnsupportedOperationException e) {
      }
  }
}

class director_exception_MyFoo extends Foo {
    public String ping() {
        throw new UnsupportedOperationException("Foo::ping not implemented");
    }
}

