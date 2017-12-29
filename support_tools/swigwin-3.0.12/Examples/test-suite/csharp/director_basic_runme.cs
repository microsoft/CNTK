using System;

namespace director_basicNamespace {

public class runme
{
  static void Main() 
  {
    runme r = new runme();
    r.run();
  }

  void run()
  {
    director_basic_MyFoo a = new director_basic_MyFoo();

    if (a.ping() != "director_basic_MyFoo::ping()") {
      throw new Exception ( "a.ping()" );
    }

    if (a.pong() != "Foo::pong();director_basic_MyFoo::ping()") {
      throw new Exception ( "a.pong()" );
    }

    Foo b = new Foo();

    if (b.ping() != "Foo::ping()") {
      throw new Exception ( "b.ping()" );
    }

    if (b.pong() != "Foo::pong();Foo::ping()") {
      throw new Exception ( "b.pong()" );
    }

    A1 a1 = new A1(1, false);
    a1.Dispose();

    {
      MyOverriddenClass my = new MyOverriddenClass();

      my.expectNull = true;
      if (MyClass.call_pmethod(my, null) != null)
        throw new Exception("null pointer marshalling problem");

      Bar myBar = new Bar();
      my.expectNull = false;
      Bar myNewBar = MyClass.call_pmethod(my, myBar);
      if (myNewBar == null)
        throw new Exception("non-null pointer marshalling problem");
      myNewBar.x = 10;
    }
  }
}

class director_basic_MyFoo : Foo {
  public director_basic_MyFoo() : base() {
  }

  public override string ping() {
    return "director_basic_MyFoo::ping()";
  }
}

class MyOverriddenClass : MyClass {
  public bool expectNull = false;
  public bool nonNullReceived = false;
  public override Bar pmethod(Bar b) {
    if ( expectNull && (b != null) )
      throw new Exception("null not received as expected");
    return b;
  }
}

}
