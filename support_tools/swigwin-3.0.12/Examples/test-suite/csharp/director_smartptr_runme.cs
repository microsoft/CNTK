using director_smartptrNamespace;
using System;

public class runme
{
  private class director_smartptr_MyBarFoo : Foo
  {
    public override string ping()
    {
      return "director_smartptr_MyBarFoo.ping()";
    }

    public override string pong()
    {
      return "director_smartptr_MyBarFoo.pong();" + ping();
    }

    public override string upcall(FooBar fooBarPtr)
    {
      return "override;" + fooBarPtr.FooBarDo();
    }

    public override Foo makeFoo()
    {
      return new Foo();
    }
  }

  private static void check(string got, string expected)
  {
    if (got != expected)
      throw new ApplicationException("Failed, got: " + got + " expected: " + expected);
  }

  static void Main()
  {
    FooBar fooBar = new FooBar();

    Foo myBarFoo = new director_smartptr_MyBarFoo();
    check(myBarFoo.ping(), "director_smartptr_MyBarFoo.ping()");
    check(Foo.callPong(myBarFoo), "director_smartptr_MyBarFoo.pong();director_smartptr_MyBarFoo.ping()");
    check(Foo.callUpcall(myBarFoo, fooBar), "override;Bar::Foo2::Foo2Bar()");

    Foo myFoo = myBarFoo.makeFoo();
    check(myFoo.pong(), "Foo::pong();Foo::ping()");
    check(Foo.callPong(myFoo), "Foo::pong();Foo::ping()");
    check(myFoo.upcall(new FooBar()), "Bar::Foo2::Foo2Bar()");

    Foo myFoo2 = new Foo().makeFoo();
    check(myFoo2.pong(), "Foo::pong();Foo::ping()");
    check(Foo.callPong(myFoo2), "Foo::pong();Foo::ping()");
  }
}
