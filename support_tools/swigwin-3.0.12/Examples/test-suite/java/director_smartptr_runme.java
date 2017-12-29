// Make sure that directors are connected and disconnected when used inconjunction with
// being a smart pointer

public class director_smartptr_runme {

  static {
    try {
      System.loadLibrary("director_smartptr");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  private static void check(String got, String expected) {
    if (!got.equals(expected))
      throw new RuntimeException("Failed, got: " + got + " expected: " + expected);
  }

  public static void main(String argv[]) {
    director_smartptr.FooBar fooBar = new director_smartptr.FooBar();

    director_smartptr.Foo myBarFoo = new director_smartptr_MyBarFoo();
    check(myBarFoo.ping(), "director_smartptr_MyBarFoo.ping()");
    check(director_smartptr.Foo.callPong(myBarFoo), "director_smartptr_MyBarFoo.pong();director_smartptr_MyBarFoo.ping()");
    check(director_smartptr.Foo.callUpcall(myBarFoo, fooBar), "override;Bar::Foo2::Foo2Bar()");

    director_smartptr.Foo myFoo = myBarFoo.makeFoo();
    check(myFoo.pong(), "Foo::pong();Foo::ping()");
    check(director_smartptr.Foo.callPong(myFoo), "Foo::pong();Foo::ping()");
    check(myFoo.upcall(fooBar), "Bar::Foo2::Foo2Bar()");

    director_smartptr.Foo myFoo2 = new director_smartptr.Foo().makeFoo();
    check(myFoo2.pong(), "Foo::pong();Foo::ping()");
    check(director_smartptr.Foo.callPong(myFoo2), "Foo::pong();Foo::ping()");
  }
}

class director_smartptr_MyBarFoo extends director_smartptr.Foo {

  @Override
  public String ping() {
    return "director_smartptr_MyBarFoo.ping()";
  }

  @Override
  public String pong() {
    return "director_smartptr_MyBarFoo.pong();" + ping();
  }

  @Override
  public String upcall(director_smartptr.FooBar fooBarPtr) {
    return "override;" + fooBarPtr.FooBarDo();
  }

  @Override
  public director_smartptr.Foo makeFoo() {
    return new director_smartptr.Foo();
  }
}
