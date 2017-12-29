
import director_default.*;

public class director_default_runme {

  static {
    try {
      System.loadLibrary("director_default");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) {
    {
      director_default_MyFoo a = new director_default_MyFoo();
      a = new director_default_MyFoo(10);
    }

    director_default_MyFoo a = new director_default_MyFoo();
    if (!a.GetMsg().equals("director_default_MyFoo-default")) {
      throw new RuntimeException ( "Test 1 failed" );
    }
    if (!a.GetMsg("boo").equals("director_default_MyFoo-boo")) {
      throw new RuntimeException ( "Test 2 failed" );
    }

    Foo b = new Foo();
    if (!b.GetMsg().equals("Foo-default")) {
      throw new RuntimeException ( "Test 1 failed" );
    }
    if (!b.GetMsg("boo").equals("Foo-boo")) {
      throw new RuntimeException ( "Test 2 failed" );
    }

  }
}

class director_default_MyFoo extends Foo {
    public director_default_MyFoo() {
      super();
    }
    public director_default_MyFoo(int i) {
      super(i);
    }
    public String Msg(String msg) { 
      return "director_default_MyFoo-" + msg; 
    }
}

