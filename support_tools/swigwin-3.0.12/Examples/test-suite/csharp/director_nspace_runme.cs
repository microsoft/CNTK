
using System;
using director_nspaceNamespace;

public class bools_runme {

  public static void Main() {
  }
}

class director_nspace_MyBarFoo : director_nspaceNamespace.TopLevel.Bar.Foo {

  public override String ping() {
    return "director_nspace_MyBarFoo.ping();";
  }

  public override String pong() {
    return "director_nspace_MyBarFoo.pong();" + ping();
  }

  public override String fooBar(director_nspaceNamespace.TopLevel.Bar.FooBar fooBar) {
    return fooBar.FooBarDo();
  }

  public override director_nspaceNamespace.TopLevel.Bar.Foo makeFoo() {
    return new director_nspaceNamespace.TopLevel.Bar.Foo();
  }

  public override director_nspaceNamespace.TopLevel.Bar.FooBar makeFooBar() {
    return new director_nspaceNamespace.TopLevel.Bar.FooBar();
  }
}
