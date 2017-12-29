using System;
using li_boost_shared_ptrNamespace;

public class runme
{
  // Debugging flag
  public static bool debug = false;

  private static void WaitForGC()
  {
    System.GC.Collect(); 
    System.GC.WaitForPendingFinalizers();
    System.Threading.Thread.Sleep(10);
  }

  static void Main() 
  {
    if (debug)
      Console.WriteLine("Started");

    li_boost_shared_ptr.debug_shared=debug;

    // Change loop count to run for a long time to monitor memory
    const int loopCount = 1; //50000;
    for (int i=0; i<loopCount; i++) {
      new runme().runtest();
      System.GC.Collect(); 
      System.GC.WaitForPendingFinalizers();
      if (i%100 == 0) {
        System.Threading.Thread.Sleep(1); // give some time to the lower priority finalizer thread
      }
    }

    if (debug)
      Console.WriteLine("Nearly finished");

    {
      int countdown = 500;
      int expectedCount = 1;
      while (true) {
        WaitForGC();
        if (--countdown == 0)
          break;
        if (Klass.getTotal_count() == expectedCount) // Expect the one global variable (GlobalValue)
          break;
      }
      int actualCount = Klass.getTotal_count();
      if (actualCount != expectedCount)
        Console.Error.WriteLine("Expected count: " + expectedCount + " Actual count: " + actualCount); // Finalizers are not guaranteed to be run and sometimes they just don't
    }

    int wrapper_count = li_boost_shared_ptr.shared_ptr_wrapper_count(); 
    if (wrapper_count != li_boost_shared_ptr.NOT_COUNTING)
      if (wrapper_count != 1) // Expect the one global variable (GlobalSmartValue)
        throw new ApplicationException("shared_ptr wrapper count=" + wrapper_count);

    if (debug)
      Console.WriteLine("Finished");
  }

  private void runtest() {
    // simple shared_ptr usage - created in C++
    {
      Klass k = new Klass("me oh my");
      String val = k.getValue();
      verifyValue("me oh my", val);
      verifyCount(1, k);
    }

    // simple shared_ptr usage - not created in C++
    {
      Klass k = li_boost_shared_ptr.factorycreate();
      String val = k.getValue();
      verifyValue("factorycreate", val);
      verifyCount(1, k);
    }

    // pass by shared_ptr
    {
      Klass k = new Klass("me oh my");
      Klass kret = li_boost_shared_ptr.smartpointertest(k);
      String val = kret.getValue();
      verifyValue("me oh my smartpointertest", val);
      verifyCount(2, k);
      verifyCount(2, kret);
    }

    // pass by shared_ptr pointer
    {
      Klass k = new Klass("me oh my");
      Klass kret = li_boost_shared_ptr.smartpointerpointertest(k);
      String val = kret.getValue();
      verifyValue("me oh my smartpointerpointertest", val);
      verifyCount(2, k);
      verifyCount(2, kret);
    }

    // pass by shared_ptr reference
    {
      Klass k = new Klass("me oh my");
      Klass kret = li_boost_shared_ptr.smartpointerreftest(k);
      String val = kret.getValue();
      verifyValue("me oh my smartpointerreftest", val);
      verifyCount(2, k);
      verifyCount(2, kret);
    }

    // pass by shared_ptr pointer reference
    {
      Klass k = new Klass("me oh my");
      Klass kret = li_boost_shared_ptr.smartpointerpointerreftest(k);
      String val = kret.getValue();
      verifyValue("me oh my smartpointerpointerreftest", val);
      verifyCount(2, k);
      verifyCount(2, kret);
    }

    // const pass by shared_ptr
    {
      Klass k = new Klass("me oh my");
      Klass kret = li_boost_shared_ptr.constsmartpointertest(k);
      String val = kret.getValue();
      verifyValue("me oh my", val);
      verifyCount(2, k);
      verifyCount(2, kret);
    }

    // const pass by shared_ptr pointer
    {
      Klass k = new Klass("me oh my");
      Klass kret = li_boost_shared_ptr.constsmartpointerpointertest(k);
      String val = kret.getValue();
      verifyValue("me oh my", val);
      verifyCount(2, k);
      verifyCount(2, kret);
    }

    // const pass by shared_ptr reference
    {
      Klass k = new Klass("me oh my");
      Klass kret = li_boost_shared_ptr.constsmartpointerreftest(k);
      String val = kret.getValue();
      verifyValue("me oh my", val);
      verifyCount(2, k);
      verifyCount(2, kret);
    }

    // pass by value
    {
      Klass k = new Klass("me oh my");
      Klass kret = li_boost_shared_ptr.valuetest(k);
      String val = kret.getValue();
      verifyValue("me oh my valuetest", val);
      verifyCount(1, k);
      verifyCount(1, kret);
    }

    // pass by pointer
    {
      Klass k = new Klass("me oh my");
      Klass kret = li_boost_shared_ptr.pointertest(k);
      String val = kret.getValue();
      verifyValue("me oh my pointertest", val);
      verifyCount(1, k);
      verifyCount(1, kret);
    }

    // pass by reference
    {
      Klass k = new Klass("me oh my");
      Klass kret = li_boost_shared_ptr.reftest(k);
      String val = kret.getValue();
      verifyValue("me oh my reftest", val);
      verifyCount(1, k);
      verifyCount(1, kret);
    }

    // pass by pointer reference
    {
      Klass k = new Klass("me oh my");
      Klass kret = li_boost_shared_ptr.pointerreftest(k);
      String val = kret.getValue();
      verifyValue("me oh my pointerreftest", val);
      verifyCount(1, k);
      verifyCount(1, kret);
    }

    // null tests
    {
      Klass k = null;

      // TODO: add in const versions too
      if (li_boost_shared_ptr.smartpointertest(k) != null)
        throw new ApplicationException("return was not null");

      if (li_boost_shared_ptr.smartpointerpointertest(k) != null)
        throw new ApplicationException("return was not null");

      if (li_boost_shared_ptr.smartpointerreftest(k) != null)
        throw new ApplicationException("return was not null");

      if (li_boost_shared_ptr.smartpointerpointerreftest(k) != null)
        throw new ApplicationException("return was not null");

      if (li_boost_shared_ptr.nullsmartpointerpointertest(null) != "null pointer")
        throw new ApplicationException("not null smartpointer pointer");

      try { li_boost_shared_ptr.valuetest(k); throw new ApplicationException("Failed to catch null pointer"); } catch (ArgumentNullException) {}

      if (li_boost_shared_ptr.pointertest(k) != null)
        throw new ApplicationException("return was not null");

      try { li_boost_shared_ptr.reftest(k); throw new ApplicationException("Failed to catch null pointer"); } catch (ArgumentNullException) {}
    }

    // $owner
    {
      Klass k = li_boost_shared_ptr.pointerownertest();
      String val = k.getValue();
      verifyValue("pointerownertest", val);
      verifyCount(1, k);
    }
    {
      Klass k = li_boost_shared_ptr.smartpointerpointerownertest();
      String val = k.getValue();
      verifyValue("smartpointerpointerownertest", val);
      verifyCount(1, k);
    }

    ////////////////////////////////// Derived classes ////////////////////////////////////////
    // derived pass by shared_ptr
    {
      KlassDerived k = new KlassDerived("me oh my");
      KlassDerived kret = li_boost_shared_ptr.derivedsmartptrtest(k);
      String val = kret.getValue();
      verifyValue("me oh my derivedsmartptrtest-Derived", val);
      verifyCount(4, k); // includes two extra references for upcasts in the proxy classes
      verifyCount(4, kret);
    }
    // derived pass by shared_ptr pointer
    {
      KlassDerived k = new KlassDerived("me oh my");
      KlassDerived kret = li_boost_shared_ptr.derivedsmartptrpointertest(k);
      String val = kret.getValue();
      verifyValue("me oh my derivedsmartptrpointertest-Derived", val);
      verifyCount(4, k); // includes two extra references for upcasts in the proxy classes
      verifyCount(4, kret);
    }
    // derived pass by shared_ptr ref
    {
      KlassDerived k = new KlassDerived("me oh my");
      KlassDerived kret = li_boost_shared_ptr.derivedsmartptrreftest(k);
      String val = kret.getValue();
      verifyValue("me oh my derivedsmartptrreftest-Derived", val);
      verifyCount(4, k); // includes two extra references for upcasts in the proxy classes
      verifyCount(4, kret);
    }
    // derived pass by shared_ptr pointer ref
    {
      KlassDerived k = new KlassDerived("me oh my");
      KlassDerived kret = li_boost_shared_ptr.derivedsmartptrpointerreftest(k);
      String val = kret.getValue();
      verifyValue("me oh my derivedsmartptrpointerreftest-Derived", val);
      verifyCount(4, k); // includes two extra references for upcasts in the proxy classes
      verifyCount(4, kret);
    }
    // derived pass by pointer
    {
      KlassDerived k = new KlassDerived("me oh my");
      KlassDerived kret = li_boost_shared_ptr.derivedpointertest(k);
      String val = kret.getValue();
      verifyValue("me oh my derivedpointertest-Derived", val);
      verifyCount(2, k); // includes an extra reference for the upcast in the proxy class
      verifyCount(2, kret);
    }
    // derived pass by ref
    {
      KlassDerived k = new KlassDerived("me oh my");
      KlassDerived kret = li_boost_shared_ptr.derivedreftest(k);
      String val = kret.getValue();
      verifyValue("me oh my derivedreftest-Derived", val);
      verifyCount(2, k); // includes an extra reference for the upcast in the proxy class
      verifyCount(2, kret);
    }

    ////////////////////////////////// Derived and base class mixed ////////////////////////////////////////
    // pass by shared_ptr (mixed)
    {
      Klass k = new KlassDerived("me oh my");
      Klass kret = li_boost_shared_ptr.smartpointertest(k);
      String val = kret.getValue();
      verifyValue("me oh my smartpointertest-Derived", val);
      verifyCount(3, k); // an extra reference for the upcast in the proxy class
      verifyCount(3, kret);
    }

    // pass by shared_ptr pointer (mixed)
    {
      Klass k = new KlassDerived("me oh my");
      Klass kret = li_boost_shared_ptr.smartpointerpointertest(k);
      String val = kret.getValue();
      verifyValue("me oh my smartpointerpointertest-Derived", val);
      verifyCount(3, k); // an extra reference for the upcast in the proxy class
      verifyCount(3, kret);
    }

    // pass by shared_ptr reference (mixed)
    {
      Klass k = new KlassDerived("me oh my");
      Klass kret = li_boost_shared_ptr.smartpointerreftest(k);
      String val = kret.getValue();
      verifyValue("me oh my smartpointerreftest-Derived", val);
      verifyCount(3, k); // an extra reference for the upcast in the proxy class
      verifyCount(3, kret);
    }

    // pass by shared_ptr pointer reference (mixed)
    {
      Klass k = new KlassDerived("me oh my");
      Klass kret = li_boost_shared_ptr.smartpointerpointerreftest(k);
      String val = kret.getValue();
      verifyValue("me oh my smartpointerpointerreftest-Derived", val);
      verifyCount(3, k); // an extra reference for the upcast in the proxy class
      verifyCount(3, kret);
    }

    // pass by value (mixed)
    {
      Klass k = new KlassDerived("me oh my");
      Klass kret = li_boost_shared_ptr.valuetest(k);
      String val = kret.getValue();
      verifyValue("me oh my valuetest", val); // note slicing
      verifyCount(2, k); // an extra reference for the upcast in the proxy class
      verifyCount(1, kret);
    }

    // pass by pointer (mixed)
    {
      Klass k = new KlassDerived("me oh my");
      Klass kret = li_boost_shared_ptr.pointertest(k);
      String val = kret.getValue();
      verifyValue("me oh my pointertest-Derived", val);
      verifyCount(2, k); // an extra reference for the upcast in the proxy class
      verifyCount(1, kret);
    }

    // pass by ref (mixed)
    {
      Klass k = new KlassDerived("me oh my");
      Klass kret = li_boost_shared_ptr.reftest(k);
      String val = kret.getValue();
      verifyValue("me oh my reftest-Derived", val);
      verifyCount(2, k); // an extra reference for the upcast in the proxy class
      verifyCount(1, kret);
    }

    // 3rd derived class
    {
      Klass k = new Klass3rdDerived("me oh my");
      String val = k.getValue();
      verifyValue("me oh my-3rdDerived", val);
      verifyCount(3, k); // 3 classes in inheritance chain == 3 swigCPtr values
      val = li_boost_shared_ptr.test3rdupcast(k);
      verifyValue("me oh my-3rdDerived", val);
      verifyCount(3, k);
    }

    ////////////////////////////////// Member variables ////////////////////////////////////////
    // smart pointer by value
    {
      MemberVariables m = new MemberVariables();
      Klass k = new Klass("smart member value");
      m.SmartMemberValue = k;
      String val = k.getValue();
      verifyValue("smart member value", val);
      verifyCount(2, k);

      Klass kmember = m.SmartMemberValue;
      val = kmember.getValue();
      verifyValue("smart member value", val);
      verifyCount(3, kmember);
      verifyCount(3, k);

      m.Dispose();
      verifyCount(2, kmember);
      verifyCount(2, k);
    }
    // smart pointer by pointer
    {
      MemberVariables m = new MemberVariables();
      Klass k = new Klass("smart member pointer");
      m.SmartMemberPointer = k;
      String val = k.getValue();
      verifyValue("smart member pointer", val);
      verifyCount(1, k);

      Klass kmember = m.SmartMemberPointer;
      val = kmember.getValue();
      verifyValue("smart member pointer", val);
      verifyCount(2, kmember);
      verifyCount(2, k);

      m.Dispose();
      verifyCount(2, kmember);
      verifyCount(2, k);
    }
    // smart pointer by reference
    {
      MemberVariables m = new MemberVariables();
      Klass k = new Klass("smart member reference");
      m.SmartMemberReference = k;
      String val = k.getValue();
      verifyValue("smart member reference", val);
      verifyCount(2, k);

      Klass kmember = m.SmartMemberReference;
      val = kmember.getValue();
      verifyValue("smart member reference", val);
      verifyCount(3, kmember);
      verifyCount(3, k);

      // The C++ reference refers to SmartMemberValue...
      Klass kmemberVal = m.SmartMemberValue;
      val = kmember.getValue();
      verifyValue("smart member reference", val);
      verifyCount(4, kmemberVal);
      verifyCount(4, kmember);
      verifyCount(4, k);

      m.Dispose();
      verifyCount(3, kmember);
      verifyCount(3, k);
    }
    // plain by value
    {
      MemberVariables m = new MemberVariables();
      Klass k = new Klass("plain member value");
      m.MemberValue = k;
      String val = k.getValue();
      verifyValue("plain member value", val);
      verifyCount(1, k);

      Klass kmember = m.MemberValue;
      val = kmember.getValue();
      verifyValue("plain member value", val);
      verifyCount(1, kmember);
      verifyCount(1, k);

      m.Dispose();
      verifyCount(1, kmember);
      verifyCount(1, k);
    }
    // plain by pointer
    {
      MemberVariables m = new MemberVariables();
      Klass k = new Klass("plain member pointer");
      m.MemberPointer = k;
      String val = k.getValue();
      verifyValue("plain member pointer", val);
      verifyCount(1, k);

      Klass kmember = m.MemberPointer;
      val = kmember.getValue();
      verifyValue("plain member pointer", val);
      verifyCount(1, kmember);
      verifyCount(1, k);

      m.Dispose();
      verifyCount(1, kmember);
      verifyCount(1, k);
    }
    // plain by reference
    {
      MemberVariables m = new MemberVariables();
      Klass k = new Klass("plain member reference");
      m.MemberReference = k;
      String val = k.getValue();
      verifyValue("plain member reference", val);
      verifyCount(1, k);

      Klass kmember = m.MemberReference;
      val = kmember.getValue();
      verifyValue("plain member reference", val);
      verifyCount(1, kmember);
      verifyCount(1, k);

      m.Dispose();
      verifyCount(1, kmember);
      verifyCount(1, k);
    }

    // null member variables
    {
      MemberVariables m = new MemberVariables();

      // shared_ptr by value
      Klass k = m.SmartMemberValue;
      if (k != null)
        throw new ApplicationException("expected null");
      m.SmartMemberValue = null;
      k = m.SmartMemberValue;
      if (k != null)
        throw new ApplicationException("expected null");
      verifyCount(0, k);

      // plain by value
      try { m.MemberValue = null; throw new ApplicationException("Failed to catch null pointer"); } catch (ArgumentNullException) {}
    }

    ////////////////////////////////// Global variables ////////////////////////////////////////
    // smart pointer
    {
      Klass kglobal = li_boost_shared_ptr.GlobalSmartValue;
      if (kglobal != null)
        throw new ApplicationException("expected null");

      Klass k = new Klass("smart global value");
      li_boost_shared_ptr.GlobalSmartValue = k;
      verifyCount(2, k);

      kglobal = li_boost_shared_ptr.GlobalSmartValue;
      String val = kglobal.getValue();
      verifyValue("smart global value", val);
      verifyCount(3, kglobal);
      verifyCount(3, k);
      verifyValue("smart global value", li_boost_shared_ptr.GlobalSmartValue.getValue());
      li_boost_shared_ptr.GlobalSmartValue = null;
    }
    // plain value
    {
      Klass kglobal;

      Klass k = new Klass("global value");
      li_boost_shared_ptr.GlobalValue = k;
      verifyCount(1, k);

      kglobal = li_boost_shared_ptr.GlobalValue;
      String val = kglobal.getValue();
      verifyValue("global value", val);
      verifyCount(1, kglobal);
      verifyCount(1, k);
      verifyValue("global value", li_boost_shared_ptr.GlobalValue.getValue());

      try { li_boost_shared_ptr.GlobalValue = null; throw new ApplicationException("Failed to catch null pointer"); } catch (ArgumentNullException) {}
    }
    // plain pointer
    {
      Klass kglobal = li_boost_shared_ptr.GlobalPointer;
      if (kglobal != null)
        throw new ApplicationException("expected null");

      Klass k = new Klass("global pointer");
      li_boost_shared_ptr.GlobalPointer = k;
      verifyCount(1, k);

      kglobal = li_boost_shared_ptr.GlobalPointer;
      String val = kglobal.getValue();
      verifyValue("global pointer", val);
      verifyCount(1, kglobal);
      verifyCount(1, k);
      li_boost_shared_ptr.GlobalPointer = null;
    }
    // plain reference
    {
      Klass kglobal;

      Klass k = new Klass("global reference");
      li_boost_shared_ptr.GlobalReference = k;
      verifyCount(1, k);

      kglobal = li_boost_shared_ptr.GlobalReference;
      String val = kglobal.getValue();
      verifyValue("global reference", val);
      verifyCount(1, kglobal);
      verifyCount(1, k);

      try { li_boost_shared_ptr.GlobalReference = null; throw new ApplicationException("Failed to catch null pointer"); } catch (ArgumentNullException) {}
    }

    ////////////////////////////////// Templates ////////////////////////////////////////
    {
      PairIntDouble pid = new PairIntDouble(10, 20.2);
      if (pid.baseVal1 != 20 || pid.baseVal2 != 40.4)
        throw new ApplicationException("Base values wrong");
      if (pid.val1 != 10 || pid.val2 != 20.2)
        throw new ApplicationException("Derived Values wrong");
    }
  }
  private void verifyValue(String expected, String got) {
    if (expected != got)
      throw new Exception("verify value failed. Expected: " + expected + " Got: " + got);
  }
  private void verifyCount(int expected, Klass k) {
    int got = li_boost_shared_ptr.use_count(k); 
    if (expected != got)
      throw new Exception("verify use_count failed. Expected: " + expected + " Got: " + got);
  }
}
