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
the overloaded Java methods, see DefaultParms method.

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
JavaDerived - Val(444.555)
JavaDerived - Ref(444.555)
JavaDerived - Ptr(444.555)
JavaDerived - FullyOverloaded(int 10)
JavaDerived - FullyOverloaded(bool True)
JavaDerived - SemiOverloaded(-678)
Base - SemiOverloaded(bool 1)
JavaDerived - DefaultParms(10, 2.2)
JavaDerived - DefaultParms(10, 1.1)
------------ Finish ------------
*/


import director_classes.*;

public class director_classes_runme {

  static {
    try {
        System.loadLibrary("director_classes");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) throws Throwable
  {
    director_classes_runme r = new director_classes_runme();
    r.run();
  }

  void run()
  {
    if (director_classes.getPrintDebug()) System.out.println("------------ Start ------------ ");

    Caller myCaller = new Caller();

    // test C++ base class
    {
      Base myBase = new Base(100.0);
      makeCalls(myCaller, myBase);
      myBase.delete();
    }

    if (director_classes.getPrintDebug()) System.out.println("--------------------------------");

    // test vanilla C++ wrapped derived class
    {
      Base myBase = new Derived(200.0);
      makeCalls(myCaller, myBase);
      myBase.delete();
    }

    if (director_classes.getPrintDebug()) System.out.println("--------------------------------");

    // test director / Java derived class
    {
      Base myBase = new JavaDerived(300.0);
      makeCalls(myCaller, myBase);
      myBase.delete();
    }

    if (director_classes.getPrintDebug()) System.out.println("------------ Finish ------------ ");
  }

  void makeCalls(Caller myCaller, Base myBase)
  {
    String baseSimpleName = getSimpleName(myBase.getClass());

    myCaller.set(myBase);

    DoubleHolder dh = new DoubleHolder(444.555);

    // Class pointer, reference and pass by value tests
    if (myCaller.ValCall(dh).getVal() != dh.getVal()) throw new RuntimeException("failed");
    if (myCaller.RefCall(dh).getVal() != dh.getVal()) throw new RuntimeException("failed");
    if (myCaller.PtrCall(dh).getVal() != dh.getVal()) throw new RuntimeException("failed");

    // Fully overloaded method test (all methods in base class are overloaded)
    if (!myCaller.FullyOverloadedCall(10).equals(baseSimpleName + "::FullyOverloaded(int)")) {
      System.out.println(myCaller.FullyOverloadedCall(10) + "----" + (baseSimpleName + "::FullyOverloaded(int)"));
      throw new RuntimeException("failed");
    }
    if (!myCaller.FullyOverloadedCall(true).equals(baseSimpleName + "::FullyOverloaded(bool)")) throw new RuntimeException("failed");

    // Semi overloaded method test (some methods in base class are overloaded)
    if (!myCaller.SemiOverloadedCall(-678).equals(baseSimpleName + "::SemiOverloaded(int)")) throw new RuntimeException("failed");
    if (!myCaller.SemiOverloadedCall(true).equals("Base" + "::SemiOverloaded(bool)")) throw new RuntimeException("failed");

    // Default parameters methods test
    if (!(myCaller.DefaultParmsCall(10, 2.2)).equals(baseSimpleName + "::DefaultParms(int, double)")) throw new RuntimeException("failed");
    if (myBase instanceof JavaDerived) { // special handling for Java derived classes, there is no way to do this any other way
      if (!myCaller.DefaultParmsCall(10).equals(baseSimpleName + "::DefaultParms(int, double)")) throw new RuntimeException("failed");
    } else {
      if (!myCaller.DefaultParmsCall(10).equals(baseSimpleName + "::DefaultParms(int)")) throw new RuntimeException("failed");
    }

    myCaller.reset();
  }

  // Same as Class.getSimpleName() which is not present in all jdks
  static String getSimpleName(Class klass) {
    String fullName = klass.getName();
    Package packag = klass.getPackage();
    String simpleName = null;
    if (packag != null)
        simpleName = fullName.replaceAll(packag.getName() + "\\.", "");
    else
        simpleName = fullName;
    return simpleName;
  }
}


class JavaDerived extends Base
{
  public JavaDerived(double dd)
  {
    super(dd);
  }

  public DoubleHolder Val(DoubleHolder x)
  {
    if (director_classes.getPrintDebug()) System.out.println("JavaDerived - Val(" + x.getVal() + ")");
    return x;
  }
  public DoubleHolder Ref(DoubleHolder x)
  {
    if (director_classes.getPrintDebug()) System.out.println("JavaDerived - Ref(" + x.getVal() + ")");
    return x;
  }
  public DoubleHolder Ptr(DoubleHolder x)
  {
    if (director_classes.getPrintDebug()) System.out.println("JavaDerived - Ptr(" + x.getVal() + ")");
    return x;
  }
  public String FullyOverloaded(int x)
  {
    if (director_classes.getPrintDebug()) System.out.println("JavaDerived - FullyOverloaded(int " + x + ")");
    return "JavaDerived::FullyOverloaded(int)";
  }
  public String FullyOverloaded(boolean x)
  {
    if (director_classes.getPrintDebug()) System.out.println("JavaDerived - FullyOverloaded(bool " + x + ")");
    return "JavaDerived::FullyOverloaded(bool)";
  }
  // Note no SemiOverloaded(bool x) method
  public String SemiOverloaded(int x)
  {
    String ret = "JavaDerived::SemiOverloaded(int)";
    if (director_classes.getPrintDebug()) System.out.println("JavaDerived - SemiOverloaded(" + x + ")");
    return ret;
  }
  public String DefaultParms(int x, double y)
  {
    String ret = "JavaDerived::DefaultParms(int, double)";
    if (director_classes.getPrintDebug()) System.out.println("JavaDerived - DefaultParms(" + x + ", " + y + ")");
    return ret;
  }
  // Note the following method can never be called from unmanaged code.
  // It is here only for code that calls it directly from managed code.
  // But should always be defined to ensure behaviour is consistent
  // independent of where DefaultParsms is called from (managed or unmanaged code).
  // Note this method can never be called from unmanaged code
  public String DefaultParms(int x)
  {
    if (director_classes.getPrintDebug()) System.out.println("JavaDerived - DefaultParms(" + x + ")");
    return DefaultParms(x, 1.1/*use C++ default here*/);
  }
}

