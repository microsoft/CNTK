
import director_protected.*;
import java.lang.reflect.*;

public class director_protected_runme {

  static {
    try {
      System.loadLibrary("director_protected");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) {

    Bar b = new Bar();
    Foo f = b.create();
    director_protected_FooBar fb = new director_protected_FooBar();
    director_protected_FooBar2 fb2 = new director_protected_FooBar2();
    director_protected_FooBar3 fb3 = new director_protected_FooBar3();

    {
      String s = fb.used();
      if (!s.equals("Foo::pang();Bar::pong();Foo::pong();director_protected_FooBar::ping();"))
        throw new RuntimeException( "bad director_protected_FooBar::used" );
    }

    {
      String s = fb2.used();
      if (!s.equals("director_protected_FooBar2::pang();Bar::pong();Foo::pong();director_protected_FooBar2::ping();"))
        throw new RuntimeException( "bad director_protected_FooBar2::used" );
    }

    {
      String s = b.pong();
      if (!s.equals("Bar::pong();Foo::pong();Bar::ping();"))
        throw new RuntimeException( "bad Bar::pong" );
    }

    {
      String s = f.pong();
      if (!s.equals("Bar::pong();Foo::pong();Bar::ping();"))
        throw new RuntimeException(" bad Foo::pong" );
    }

    {
      String s3 = fb.pong();
      if (!s3.equals("Bar::pong();Foo::pong();director_protected_FooBar::ping();"))
        throw new RuntimeException(" bad director_protected_FooBar::pong" );
    }

    try {

      Method method = b.getClass().getDeclaredMethod("ping", (java.lang.Class[])null);
      if ( !Modifier.isProtected(method.getModifiers()) )
        throw new RuntimeException("Bar::ping should be protected" );

      method = f.getClass().getDeclaredMethod("ping", (java.lang.Class[])null);
      if ( !Modifier.isProtected(method.getModifiers()) )
        throw new RuntimeException("Foo::ping should be protected" );

      method = b.getClass().getDeclaredMethod("cheer", (java.lang.Class[])null);
      if ( !Modifier.isProtected(method.getModifiers()) )
        throw new RuntimeException("Bar::cheer should be protected" );

      method = f.getClass().getDeclaredMethod("cheer", (java.lang.Class[])null);
      if ( !Modifier.isProtected(method.getModifiers()) )
        throw new RuntimeException("Foo::cheer should be protected" );

    } catch (NoSuchMethodException n) {
      throw new RuntimeException(n);
    } catch (SecurityException s) {
      throw new RuntimeException("SecurityException caught. Test failed.");
    }

    if (!fb3.cheer().equals("director_protected_FooBar3::cheer();"))
      throw new RuntimeException("bad fb3::cheer");

    if (!fb2.callping().equals("director_protected_FooBar2::ping();"))
      throw new RuntimeException("bad fb2.callping");

    if (!fb2.callcheer().equals("director_protected_FooBar2::pang();Bar::pong();Foo::pong();director_protected_FooBar2::ping();"))
      throw new RuntimeException("bad fb2.callcheer");

    if (!fb3.callping().equals("Bar::ping();"))
      throw new RuntimeException("bad fb3.callping");

    if (!fb3.callcheer().equals("director_protected_FooBar3::cheer();"))
      throw new RuntimeException("bad fb3.callcheer");
  }
}

class director_protected_FooBar extends Bar {
  public String ping() {
    return "director_protected_FooBar::ping();";
  }
}

class director_protected_FooBar2 extends Bar {
  public String ping() {
    return "director_protected_FooBar2::ping();";
  }
  public String pang() {
    return "director_protected_FooBar2::pang();";
  }
}

class director_protected_FooBar3 extends Bar {
  public String cheer() {
    return "director_protected_FooBar3::cheer();";
  }
}

