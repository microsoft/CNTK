/*
  This test program shows a C# class CSharpDerived inheriting from Base. Three types of classes are created
  and the virtual methods called to demonstrate:
  1) Wide variety of primitive types
  2) Calling methods with zero, one or more parameters
  3) Director methods that are not overridden in C#
  4) Director classes that are not overridden at all in C#, ie non-director behaviour is as expected for director classes
  5) Inheritance hierarchy using director methods
  6) Return types working as well as parameters

  The Caller class is a tester class, which calls the virtual functions from C++.
*/

using System;
using director_primitivesNamespace;

public class runme
{
  static void Main() 
  {
    runme r = new runme();
    r.run();
  }

  void run()
  {
    if (director_primitives.PrintDebug) Console.WriteLine("------------ Start ------------ ");

    Caller myCaller = new Caller();

    // test C++ base class
    using (Base myBase = new Base(100.0))
    {
      makeCalls(myCaller, myBase);
    }

    if (director_primitives.PrintDebug) Console.WriteLine("--------------------------------");

    // test vanilla C++ wrapped derived class
    using (Base myBase = new Derived(200.0))
    {
      makeCalls(myCaller, myBase);
    }

    if (director_primitives.PrintDebug) Console.WriteLine("--------------------------------");

    // test director / C# derived class
    using (Base myBase = new CSharpDerived(300.0))
    {
      makeCalls(myCaller, myBase);
    }

    if (director_primitives.PrintDebug) Console.WriteLine("------------ Finish ------------ ");
  }

  void makeCalls(Caller myCaller, Base myBase)
  {
    myCaller.set(myBase);

    myCaller.NoParmsMethodCall();
    if (myCaller.BoolMethodCall(true) != true) throw new Exception("failed");
    if (myCaller.BoolMethodCall(false) != false) throw new Exception("failed");
    if (myCaller.IntMethodCall(-123) != -123) throw new Exception("failed");
    if (myCaller.UIntMethodCall(123) != 123) throw new Exception("failed");
    if (myCaller.FloatMethodCall((float)-123.456) != (float)-123.456) throw new Exception("failed");
    if (myCaller.CharPtrMethodCall("test string") != "test string") throw new Exception("failed");
    if (myCaller.ConstCharPtrMethodCall("another string") != "another string") throw new Exception("failed");
    if (myCaller.EnumMethodCall(HShadowMode.HShadowHard) != HShadowMode.HShadowHard) throw new Exception("failed");
    myCaller.ManyParmsMethodCall(true, -123, 123, (float)123.456, "test string", "another string", HShadowMode.HShadowHard);
    myCaller.NotOverriddenMethodCall();

    myCaller.reset();
  }
}

public class CSharpDerived : Base
{
  public CSharpDerived(double dd)
    : base(dd)
  {
  }

  public override void NoParmsMethod()
  {
    if (director_primitives.PrintDebug) Console.WriteLine("CSharpDerived - NoParmsMethod()");
  }
  public override bool BoolMethod(bool x)
  {
    if (director_primitives.PrintDebug) Console.WriteLine("CSharpDerived - BoolMethod({0})", x);
    return x;
  }
  public override int IntMethod(int x)
  {
    if (director_primitives.PrintDebug) Console.WriteLine("CSharpDerived - IntMethod({0})", x);
    return x;
  }
  public override uint UIntMethod(uint x)
  {
    if (director_primitives.PrintDebug) Console.WriteLine("CSharpDerived - UIntMethod({0})", x);
    return x;
  }
  public override float FloatMethod(float x)
  {
    if (director_primitives.PrintDebug) Console.WriteLine("CSharpDerived - FloatMethod({0})", x);
    return x;
  }
  public override string CharPtrMethod(string x)
  {
    if (director_primitives.PrintDebug) Console.WriteLine("CSharpDerived - CharPtrMethod({0})", x);
    return x;
  }
  public override string ConstCharPtrMethod(string x)
  {
    if (director_primitives.PrintDebug) Console.WriteLine("CSharpDerived - ConstCharPtrMethod({0})", x);
    return x;
  }
  public override HShadowMode EnumMethod(HShadowMode x)
  {
    if (director_primitives.PrintDebug) Console.WriteLine("CSharpDerived - EnumMethod({0})", x);
    return x;
  }
  public override void ManyParmsMethod(bool b, int i, uint u, float f, string c, string cc, HShadowMode h)
  {
    if (director_primitives.PrintDebug) Console.WriteLine("CSharpDerived - ManyParmsMethod({0}, {1}, {2}, {3}, {4}, {5}, {6})", b, i, u, f, c, cc, h);
  }
}

