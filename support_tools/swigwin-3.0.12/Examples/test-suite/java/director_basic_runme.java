
import director_basic.*;

public class director_basic_runme {

  static {
    try {
      System.loadLibrary("director_basic");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) {

    director_basic_MyFoo a = new director_basic_MyFoo();

    if (!a.ping().equals("director_basic_MyFoo::ping()")) {
      throw new RuntimeException ( "a.ping()" );
    }

    if (!a.pong().equals("Foo::pong();director_basic_MyFoo::ping()")) {
      throw new RuntimeException ( "a.pong()" );
    }

    Foo b = new Foo();

    if (!b.ping().equals("Foo::ping()")) {
      throw new RuntimeException ( "b.ping()" );
    }

    if (!b.pong().equals("Foo::pong();Foo::ping()")) {
      throw new RuntimeException ( "b.pong()" );
    }

    A1 a1 = new A1(1, false);
    a1.delete();

    {
      MyOverriddenClass my = new MyOverriddenClass();

      my.expectNull = true;
      if (MyClass.call_pmethod(my, null) != null)
        throw new RuntimeException("null pointer marshalling problem");

      Bar myBar = new Bar();
      my.expectNull = false;
      Bar myNewBar = MyClass.call_pmethod(my, myBar);
      if (myNewBar == null)
        throw new RuntimeException("non-null pointer marshalling problem");
      myNewBar.setX(10);
    }
  }
}

class director_basic_MyFoo extends Foo {
    public String ping() {
        return "director_basic_MyFoo::ping()";
    }
}

class MyOverriddenClass extends MyClass {
  public boolean expectNull = false;
  public boolean nonNullReceived = false;
  public Bar pmethod(Bar b) {
    if ( expectNull && (b != null) )
      throw new RuntimeException("null not received as expected");
    return b;
  }
}

