/*
  This test program shows a C# class JavaDerived inheriting from Base. Three types of classes are created
  and the virtual methods called to demonstrate:
  1) Wide variety of primitive types
  2) Calling methods with zero, one or more parameters
  3) Director methods that are not overridden in C#
  4) Director classes that are not overridden at all in C#, ie non-director behaviour is as expected for director classes
  5) Inheritance hierarchy using director methods
  6) Return types working as well as parameters

  The Caller class is a tester class, which calls the virtual functions from C++.
*/

import director_primitives.*;

public class director_primitives_runme {

  static {
    try {
        System.loadLibrary("director_primitives");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) throws Throwable
  {
    director_primitives_runme r = new director_primitives_runme();
    r.run();
  }

  void run()
  {
    if (director_primitives.getPrintDebug()) System.out.println("------------ Start ------------ ");

    Caller myCaller = new Caller();

    // test C++ base class
    {
      Base myBase = new Base(100.0);
      makeCalls(myCaller, myBase);
      myBase.delete();
    }

    if (director_primitives.getPrintDebug()) System.out.println("--------------------------------");

    // test vanilla C++ wrapped derived class
    {
      Base myBase = new Derived(200.0);
      makeCalls(myCaller, myBase);
      myBase.delete();
    }

    if (director_primitives.getPrintDebug()) System.out.println("--------------------------------");

    // test director / C# derived class
    {
      Base myBase = new JavaDerived(300.0);
      makeCalls(myCaller, myBase);
      myBase.delete();
    }

    if (director_primitives.getPrintDebug()) System.out.println("------------ Finish ------------ ");
  }

  void makeCalls(Caller myCaller, Base myBase)
  {
    myCaller.set(myBase);

    myCaller.NoParmsMethodCall();
    if (myCaller.BoolMethodCall(true) != true) throw new RuntimeException("failed");
    if (myCaller.BoolMethodCall(false) != false) throw new RuntimeException("failed");
    if (myCaller.IntMethodCall(-123) != -123) throw new RuntimeException("failed");
    if (myCaller.UIntMethodCall(123) != 123) throw new RuntimeException("failed");
    if (myCaller.FloatMethodCall((float)-123.456) != (float)-123.456) throw new RuntimeException("failed");
    if (!myCaller.CharPtrMethodCall("test string").equals("test string")) throw new RuntimeException("failed");
    if (!myCaller.ConstCharPtrMethodCall("another string").equals("another string")) throw new RuntimeException("failed");
    if (myCaller.EnumMethodCall(HShadowMode.HShadowHard) != HShadowMode.HShadowHard) throw new RuntimeException("failed");
    myCaller.ManyParmsMethodCall(true, -123, 123, (float)123.456, "test string", "another string", HShadowMode.HShadowHard);
    myCaller.NotOverriddenMethodCall();

    myCaller.reset();
  }
}

class JavaDerived extends Base
{
  public JavaDerived(double dd)
  {
    super(dd);
  }

  public void NoParmsMethod()
  {
    if (director_primitives.getPrintDebug()) System.out.println("JavaDerived - NoParmsMethod()");
  }
  public boolean BoolMethod(boolean x)
  {
    if (director_primitives.getPrintDebug()) System.out.println("JavaDerived - BoolMethod(" + x + ")");
    return x;
  }
  public int IntMethod(int x)
  {
    if (director_primitives.getPrintDebug()) System.out.println("JavaDerived - IntMethod(" + x + ")");
    return x;
  }
  public long UIntMethod(long x)
  {
    if (director_primitives.getPrintDebug()) System.out.println("JavaDerived - UIntMethod(" + x + ")");
    return x;
  }
  public float FloatMethod(float x)
  {
    if (director_primitives.getPrintDebug()) System.out.println("JavaDerived - FloatMethod(" + x + ")");
    return x;
  }
  public String CharPtrMethod(String x)
  {
    if (director_primitives.getPrintDebug()) System.out.println("JavaDerived - CharPtrMethod(" + x + ")");
    return x;
  }
  public String ConstCharPtrMethod(String x)
  {
    if (director_primitives.getPrintDebug()) System.out.println("JavaDerived - ConstCharPtrMethod(" + x + ")");
    return x;
  }
  public HShadowMode EnumMethod(HShadowMode x)
  {
    if (director_primitives.getPrintDebug()) System.out.println("JavaDerived - EnumMethod(" + x + ")");
    return x;
  }
  public void ManyParmsMethod(boolean b, int i, long u, float f, String c, String cc, HShadowMode h)
  {
    if (director_primitives.getPrintDebug()) System.out.println("JavaDerived - ManyParmsMethod(" + b + ", " + i + ", " + u + ", " + f + ", " + c + ", " + cc + ", " + h);
  }
}

