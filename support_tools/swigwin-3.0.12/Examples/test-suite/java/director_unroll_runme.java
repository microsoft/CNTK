
import director_unroll.*;

public class director_unroll_runme {

  static {
    try {
      System.loadLibrary("director_unroll");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) {

      director_unroll_MyFoo a = new director_unroll_MyFoo();

      Bar b = new Bar();

      b.set(a);
      Foo c = b.get();

      if (!c.ping().equals("director_unroll_MyFoo::ping()"))
          throw new RuntimeException ( "c.ping()" );
  }
}

class director_unroll_MyFoo extends Foo {
    public String ping() {
        return "director_unroll_MyFoo::ping()";
    }
}

