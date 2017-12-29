/*
This test demonstrates director classes when the types are classes.
Shown are virtual function calls which use classes passed by:
  - Value
  - Reference
  - Pointer
as both parameters and return values.
The test also demonstrates directors used with:
  - method overloading
  - default parameters
Note: Methods with default parameters that call up from C++ cannot call
the overloaded C# methods, see DefaultParms method.

Expected output if PrintDebug enabled:
------------ Start ------------
Base - Val(444.555)
Base - Ref(444.555)
Base - Ptr(444.555)
Base - FullyOverloaded(int 10)
Base - FullyOverloaded(bool 1)
Base - SemiOverloaded(int -678)
Base - SemiOverloaded(bool 1)
Base - DefaultParms(10, 2.2)
Base - DefaultParms(10, 1.1)
--------------------------------
Derived - Val(444.555)
Derived - Ref(444.555)
Derived - Ptr(444.555)
Derived - FullyOverloaded(int 10)
Derived - FullyOverloaded(bool 1)
Derived - SemiOverloaded(int -678)
Base - SemiOverloaded(bool 1)
Derived - DefaultParms(10, 2.2)
Derived - DefaultParms(10, 1.1)
--------------------------------
CSharpDerived - Val(444.555)
CSharpDerived - Ref(444.555)
CSharpDerived - Ptr(444.555)
CSharpDerived - FullyOverloaded(int 10)
CSharpDerived - FullyOverloaded(bool True)
CSharpDerived - SemiOverloaded(-678)
Base - SemiOverloaded(bool 1)
CSharpDerived - DefaultParms(10, 2.2)
CSharpDerived - DefaultParms(10, 1.1)
------------ Finish ------------
*/

using System;

namespace director_classesNamespace {

public class runme
{
  static void Main() 
  {
    runme r = new runme();
    r.run();
  }

  void run()
  {
    if (director_classes.PrintDebug) Console.WriteLine("------------ Start ------------ ");

    Caller myCaller = new Caller();

    // test C++ base class
    using (Base myBase = new Base(100.0))
    {
      makeCalls(myCaller, myBase);
    }

    if (director_classes.PrintDebug) Console.WriteLine("--------------------------------");

    // test vanilla C++ wrapped derived class
    using (Base myBase = new Derived(200.0))
    {
      makeCalls(myCaller, myBase);
    }

    if (director_classes.PrintDebug) Console.WriteLine("--------------------------------");

    // test director / C# derived class
    using (Base myBase = new CSharpDerived(300.0))
    {
      makeCalls(myCaller, myBase);
    }

    if (director_classes.PrintDebug) Console.WriteLine("------------ Finish ------------ ");
  }

  void makeCalls(Caller myCaller, Base myBase)
  {
    string NAMESPACE = "director_classesNamespace.";
    myCaller.set(myBase);

    DoubleHolder dh = new DoubleHolder(444.555);

    // Class pointer, reference and pass by value tests
    if (myCaller.ValCall(dh).val != dh.val) throw new Exception("failed");
    if (myCaller.RefCall(dh).val != dh.val) throw new Exception("failed");
    if (myCaller.PtrCall(dh).val != dh.val) throw new Exception("failed");

    // Fully overloaded method test (all methods in base class are overloaded)
    if (NAMESPACE + myCaller.FullyOverloadedCall(10) != myBase.GetType() + "::FullyOverloaded(int)") throw new Exception("failed");
    if (NAMESPACE + myCaller.FullyOverloadedCall(true) != myBase.GetType() + "::FullyOverloaded(bool)") throw new Exception("failed");

    // Semi overloaded method test (some methods in base class are overloaded)
    if (NAMESPACE + myCaller.SemiOverloadedCall(-678) != myBase.GetType() + "::SemiOverloaded(int)") throw new Exception("failed");
    if (myCaller.SemiOverloadedCall(true) != "Base" + "::SemiOverloaded(bool)") throw new Exception("failed");

    // Default parameters methods test
    if (NAMESPACE + myCaller.DefaultParmsCall(10, 2.2) != myBase.GetType() + "::DefaultParms(int, double)") throw new Exception("failed");
    if (myBase.GetType() == typeof(CSharpDerived)) { // special handling for C# derived classes, there is no way to do this any other way
      if (NAMESPACE + myCaller.DefaultParmsCall(10) != myBase.GetType() + "::DefaultParms(int, double)") throw new Exception("failed");
    } else {
      if (NAMESPACE + myCaller.DefaultParmsCall(10) != myBase.GetType() + "::DefaultParms(int)") throw new Exception("failed");
    }

    myCaller.reset();
  }
}

public class CSharpDerived : Base
{
  public CSharpDerived(double dd)
    : base(dd)
  {
  }

  public override DoubleHolder Val(DoubleHolder x)
  {
    if (director_classes.PrintDebug) Console.WriteLine("CSharpDerived - Val({0})", x.val);
    return x;
  }
  public override DoubleHolder Ref(DoubleHolder x)
  {
    if (director_classes.PrintDebug) Console.WriteLine("CSharpDerived - Ref({0})", x.val);
    return x;
  }
  public override DoubleHolder Ptr(DoubleHolder x)
  {
    if (director_classes.PrintDebug) Console.WriteLine("CSharpDerived - Ptr({0})", x.val);
    return x;
  }
  public override String FullyOverloaded(int x)
  {
    if (director_classes.PrintDebug) Console.WriteLine("CSharpDerived - FullyOverloaded(int {0})", x);
    return "CSharpDerived::FullyOverloaded(int)";
  }
  public override String FullyOverloaded(bool x)
  {
    if (director_classes.PrintDebug) Console.WriteLine("CSharpDerived - FullyOverloaded(bool {0})", x);
    return "CSharpDerived::FullyOverloaded(bool)";
  }
  // Note no SemiOverloaded(bool x) method
  public override String SemiOverloaded(int x)
  {
    String ret = "CSharpDerived::SemiOverloaded(int)";
    if (director_classes.PrintDebug) Console.WriteLine("CSharpDerived - SemiOverloaded({0})", x);
    return ret;
  }
  public override String DefaultParms(int x, double y)
  {
    String ret = "CSharpDerived::DefaultParms(int, double)";
    if (director_classes.PrintDebug) Console.WriteLine("CSharpDerived - DefaultParms({0}, {1})", x, y);
    return ret;
  }
  // Note the following method can never be called from unmanaged code.
  // It is here only for code that calls it directly from managed code.
  // But should always be defined to ensure behaviour is consistent
  // independent of where DefaultParsms is called from (managed or unmanaged code).
  // Note this method can never be called from unmanaged code
  public override String DefaultParms(int x)
  {
    if (director_classes.PrintDebug) Console.WriteLine("CSharpDerived - DefaultParms({0})", x);
    return DefaultParms(x, 1.1/*use C++ default here*/);
  }
}

}
