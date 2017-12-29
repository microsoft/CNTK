
import director_abstract.*;

public class director_abstract_runme {

  static {
    try {
      System.loadLibrary("director_abstract");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) {

      director_abstract_MyFoo a = new director_abstract_MyFoo();

      if (!a.ping().equals("director_abstract_MyFoo::ping()")) {
          throw new RuntimeException ( "a.ping()" );
      }

      if (!a.pong().equals("Foo::pong();director_abstract_MyFoo::ping()")) {
          throw new RuntimeException ( "a.pong()" );
      }

      director_abstract_BadFoo b = new director_abstract_BadFoo();
      try {
        b.ping();
        System.out.println( "Test failed. An attempt to call a pure virtual method should throw an exception" );
        System.exit(1);
      }
      catch (RuntimeException e) {
      }
  }
}

class director_abstract_MyFoo extends Foo {
    public String ping() {
        return "director_abstract_MyFoo::ping()";
    }
}

class director_abstract_BadFoo extends Foo {
}

