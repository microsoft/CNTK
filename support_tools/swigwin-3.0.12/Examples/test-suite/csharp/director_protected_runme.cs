using System;

namespace director_protectedNamespace {

public class runme
{
  static void Main() 
  {
    runme r = new runme();
    r.run();
  }

  void run()
  {
    Bar b  = new Bar();
    Foo f = b.create();
    FooBar fb = new FooBar();
    FooBar2 fb2 = new FooBar2();
    FooBar3 fb3 = new FooBar3();

    String s;
    s = fb.used();
    if ( s != ("Foo::pang();Bar::pong();Foo::pong();FooBar::ping();"))
      throw new Exception("bad FooBar::used" + " - " + s);

    s = fb2.used();
    if ( s != ("FooBar2::pang();Bar::pong();Foo::pong();FooBar2::ping();"))
      throw new Exception("bad FooBar2::used");

    s = b.pong();
    if ( s != ("Bar::pong();Foo::pong();Bar::ping();"))
      throw new Exception("bad Bar::pong");

    s = f.pong();
    if ( s != ("Bar::pong();Foo::pong();Bar::ping();"))
      throw new Exception("bad Foo::pong");

    s = fb.pong();
    if ( s != ("Bar::pong();Foo::pong();FooBar::ping();"))
      throw new Exception("bad FooBar::pong");

//    if (fb3.cheer() != "FooBar3::cheer();")
//      throw new Exception("bad fb3::cheer");

    if (fb2.callping() != "FooBar2::ping();")
      throw new Exception("bad fb2.callping");

    if (fb2.callcheer() != "FooBar2::pang();Bar::pong();Foo::pong();FooBar2::ping();")
      throw new Exception("bad fb2.callcheer");

    if (fb3.callping() != "Bar::ping();")
      throw new Exception("bad fb3.callping");

    if (fb3.callcheer() != "FooBar3::cheer();")
      throw new Exception("bad fb3.callcheer");
  }
}

class FooBar : Bar
{
  public FooBar() : base()
  {
  }

  protected override String ping()
  {
    return "FooBar::ping();";
  }
}

class FooBar2 : Bar
{
  public FooBar2() : base()
  {
  }

  protected override String ping()
  {
    return "FooBar2::ping();";
  }

  protected override String pang()
  {
    return "FooBar2::pang();";
  }
}

class FooBar3 : Bar
{
  public FooBar3() : base()
  {
  }

  protected override String cheer()
  {
    return "FooBar3::cheer();";
  }
}

}
