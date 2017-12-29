// Make sure that directors are connected and disconnected when used inconjunction with
// the %nspace feature

public class director_nspace_runme {

  static {
    try {
      System.loadLibrary("director_nspace");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) {
    director_nspace_MyBarFoo myBarFoo =
      new director_nspace_MyBarFoo();
  }

}

class director_nspace_MyBarFoo extends director_nspacePackage.TopLevel.Bar.Foo {

  @Override
  public String ping() {
    return "director_nspace_MyBarFoo.ping();";
  }

  @Override
  public String pong() {
    return "director_nspace_MyBarFoo.pong();" + ping();
  }

  @Override
  public String fooBar(director_nspacePackage.TopLevel.Bar.FooBar fooBar) {
    return fooBar.FooBarDo();
  }

  @Override
  public director_nspacePackage.TopLevel.Bar.Foo makeFoo() {
    return new director_nspacePackage.TopLevel.Bar.Foo();
  }

  @Override
  public director_nspacePackage.TopLevel.Bar.FooBar makeFooBar() {
    return new director_nspacePackage.TopLevel.Bar.FooBar();
  }
}
