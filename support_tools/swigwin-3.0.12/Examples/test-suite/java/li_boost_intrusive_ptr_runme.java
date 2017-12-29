import li_boost_intrusive_ptr.*;

public class li_boost_intrusive_ptr_runme {
  static {
    try {
        System.loadLibrary("li_boost_intrusive_ptr");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  // Debugging flag
  public final static boolean debug = false;

  private static void WaitForGC()
  {
    System.gc();
    System.runFinalization();
    try {
      java.lang.Thread.sleep(10);
    } catch (java.lang.InterruptedException e) {
    }
  }

  public static void main(String argv[])
  {
    if (debug)
      System.out.println("Started");

    li_boost_intrusive_ptr.setDebug_shared(debug);

    // Change loop count to run for a long time to monitor memory
    final int loopCount = 5000; //5000;
    for (int i=0; i<loopCount; i++) {
      new li_boost_intrusive_ptr_runme().runtest(i);
      System.gc();
      System.runFinalization();
      try {
        if (i%100 == 0) {
          java.lang.Thread.sleep(1); // give some time to the lower priority finalizer thread
        }
      } catch (java.lang.InterruptedException e) {
      }
    }

    if (debug)
      System.out.println("Nearly finished");

    int countdown = 50;
    while (true) {
      WaitForGC();
      if (--countdown == 0)
        break;
      if (Klass.getTotal_count() == 1 && KlassWithoutRefCount.getTotal_count() == 0 &&
          li_boost_intrusive_ptr.getTotal_IgnoredRefCountingBase_count() == 0 &&
          KlassDerived.getTotal_count() == 0 && KlassDerivedDerived.getTotal_count() == 1)
        // Expect 1 Klass instance - the one global variable (GlobalValue)
        break;
    }
    if (Klass.getTotal_count() != 1)
      throw new RuntimeException("Klass.total_count=" + Klass.getTotal_count());
    if (KlassWithoutRefCount.getTotal_count() != 0)
      throw new RuntimeException("KlassWithoutRefCount.total_count=" + KlassWithoutRefCount.getTotal_count());
    if (li_boost_intrusive_ptr.getTotal_IgnoredRefCountingBase_count() != 0)
      throw new RuntimeException("IgnoredRefCountingBase.total_count=" + li_boost_intrusive_ptr.getTotal_IgnoredRefCountingBase_count());
    if (KlassDerived.getTotal_count() != 0)
      throw new RuntimeException("KlassDerived.total_count=" + KlassDerived.getTotal_count());
    if (KlassDerivedDerived.getTotal_count() != 0)
      throw new RuntimeException("KlassDerivedDerived.total_count=" + KlassDerivedDerived.getTotal_count());

    int wrapper_count = li_boost_intrusive_ptr.intrusive_ptr_wrapper_count();
    if (wrapper_count != li_boost_intrusive_ptr.getNOT_COUNTING())
      if (wrapper_count != 1) // Expect 1 instance - the one global variable (GlobalSmartValue)
        throw new RuntimeException("shared_ptr wrapper count=" + wrapper_count);

    if (debug)
      System.out.println("Finished");
  }

  private int loopCount = 0;
  private void runtest(int loopCount) {
    this.loopCount = loopCount;
    // simple shared_ptr usage - created in C++
    {
      Klass k = new Klass("me oh my");
      String val = k.getValue();
      verifyValue("me oh my", val);
      verifyCount(1, k);
    }

    // simple shared_ptr usage - not created in C++
    {
      Klass k = li_boost_intrusive_ptr.factorycreate();
      String val = k.getValue();
      verifyValue("factorycreate", val);
      verifyCount(1, k);
    }

    // pass by shared_ptr
    {
      Klass k = new Klass("me oh my");
      Klass kret = li_boost_intrusive_ptr.smartpointertest(k);
      String val = kret.getValue();
      verifyValue("me oh my smartpointertest", val);
      verifyCount(1, k);
      verifyIntrusiveCount(2, k);
      verifyCount(1, kret);
      verifyIntrusiveCount(2, kret);
    }

    // pass by shared_ptr pointer
    {
      Klass k = new Klass("me oh my");
      Klass kret = li_boost_intrusive_ptr.smartpointerpointertest(k);
      String val = kret.getValue();
      verifyValue("me oh my smartpointerpointertest", val);
      verifyCount(1, k);
      verifyIntrusiveCount(2, k);
      verifyCount(1, kret);
      verifyIntrusiveCount(2, kret);
    }

    // pass by shared_ptr reference
    {
      Klass k = new Klass("me oh my");
      Klass kret = li_boost_intrusive_ptr.smartpointerreftest(k);
      String val = kret.getValue();
      verifyValue("me oh my smartpointerreftest", val);
      verifyCount(1, k);
      verifyIntrusiveCount(2, k);
      verifyCount(1, kret);
      verifyIntrusiveCount(2, kret);
    }

    // pass by shared_ptr pointer reference
    {
      Klass k = new Klass("me oh my");
      Klass kret = li_boost_intrusive_ptr.smartpointerpointerreftest(k);
      String val = kret.getValue();
      verifyValue("me oh my smartpointerpointerreftest", val);
      verifyCount(1, k);
      verifyIntrusiveCount(2, k);
      verifyCount(1, kret);
      verifyIntrusiveCount(2, kret);
    }

    // const pass by shared_ptr
    {
      Klass k = new Klass("me oh my");
      Klass kret = li_boost_intrusive_ptr.constsmartpointertest(k);
      String val = kret.getValue();
      verifyValue("me oh my", val);
      verifyCount(1, k);
      verifyIntrusiveCount(2, k);
      verifyCount(1, kret);
      verifyIntrusiveCount(2, kret);
    }

    // const pass by shared_ptr pointer
    {
      Klass k = new Klass("me oh my");
      Klass kret = li_boost_intrusive_ptr.constsmartpointerpointertest(k);
      String val = kret.getValue();
      verifyValue("me oh my", val);
      verifyCount(1, k);
      verifyIntrusiveCount(2, k);
      verifyCount(1, kret);
      verifyIntrusiveCount(2, kret);
    }

    // const pass by shared_ptr reference
    {
      Klass k = new Klass("me oh my");
      Klass kret = li_boost_intrusive_ptr.constsmartpointerreftest(k);
      String val = kret.getValue();
      verifyValue("me oh my", val);
      verifyCount(1, k);
      verifyIntrusiveCount(2, k);
      verifyCount(1, kret);
      verifyIntrusiveCount(2, kret);
    }

    // pass by value
    {
      Klass k = new Klass("me oh my");
      Klass kret = li_boost_intrusive_ptr.valuetest(k);
      String val = kret.getValue();
      verifyValue("me oh my valuetest", val);
      verifyCount(1, k);
      verifyCount(1, kret);
    }

    // pass by pointer
    {
      Klass k = new Klass("me oh my");
      Klass kret = li_boost_intrusive_ptr.pointertest(k);
      String val = kret.getValue();
      verifyValue("me oh my pointertest", val);
      verifyCount(1, k);
      verifyCount(1, kret);
    }

    // pass by reference
    {
      Klass k = new Klass("me oh my");
      Klass kret = li_boost_intrusive_ptr.reftest(k);
      String val = kret.getValue();
      verifyValue("me oh my reftest", val);
      verifyCount(1, k);
      verifyCount(1, kret);
    }

    // pass by pointer reference
    {
      Klass k = new Klass("me oh my");
      Klass kret = li_boost_intrusive_ptr.pointerreftest(k);
      String val = kret.getValue();
      verifyValue("me oh my pointerreftest", val);
      verifyCount(1, k);
      verifyCount(1, kret);
    }

    // null tests
    {
      Klass k = null;

      if (li_boost_intrusive_ptr.smartpointertest(k) != null)
        throw new RuntimeException("return was not null");

      if (li_boost_intrusive_ptr.smartpointerpointertest(k) != null)
        throw new RuntimeException("return was not null");

      if (li_boost_intrusive_ptr.smartpointerreftest(k) != null)
        throw new RuntimeException("return was not null");

      if (li_boost_intrusive_ptr.smartpointerpointerreftest(k) != null)
        throw new RuntimeException("return was not null");

      if (!li_boost_intrusive_ptr.nullsmartpointerpointertest(null).equals("null pointer"))
        throw new RuntimeException("not null smartpointer pointer");

      try { li_boost_intrusive_ptr.valuetest(k); throw new RuntimeException("Failed to catch null pointer"); } catch (NullPointerException e) {}

      if (li_boost_intrusive_ptr.pointertest(k) != null)
        throw new RuntimeException("return was not null");

      try { li_boost_intrusive_ptr.reftest(k); throw new RuntimeException("Failed to catch null pointer"); } catch (NullPointerException e) {}
    }

    // $owner
    {
      Klass k = li_boost_intrusive_ptr.pointerownertest();
      String val = k.getValue();
      verifyValue("pointerownertest", val);
      verifyCount(1, k);
    }
    {
      Klass k = li_boost_intrusive_ptr.smartpointerpointerownertest();
      String val = k.getValue();
      verifyValue("smartpointerpointerownertest", val);
      verifyCount(1, k);
    }

    ////////////////////////////////// Derived classes ////////////////////////////////////////
    // derived access to base class which cannot be wrapped in an intrusive_ptr
	{
	  KlassWithoutRefCount k = new KlassDerived("me oh my");
	  verifyValue("this class cannot be wrapped by intrusive_ptrs but we can still use it", k.getSpecialValueFromUnwrappableClass());
	}
    // derived pass by shared_ptr
    {
      KlassDerived k = new KlassDerived("me oh my");
      KlassDerived kret = li_boost_intrusive_ptr.derivedsmartptrtest(k);
      String val = kret.getValue();
      verifyValue("me oh my derivedsmartptrtest-Derived", val);
      verifyIntrusiveCount(2, k);
      verifyCount(2, k); // includes extra reference for upcast
      verifyIntrusiveCount(2, kret);
      verifyCount(2, kret);
	}

    // derived pass by shared_ptr pointer
    {
      KlassDerived k = new KlassDerived("me oh my");
      KlassDerived kret = li_boost_intrusive_ptr.derivedsmartptrpointertest(k);
      String val = kret.getValue();
      verifyValue("me oh my derivedsmartptrpointertest-Derived", val);
      verifyIntrusiveCount(2, k);
      verifyCount(2, k); // includes extra reference for upcast
      verifyIntrusiveCount(2, kret);
      verifyCount(2, kret);
    }
    // derived pass by shared_ptr ref
    {
      KlassDerived k = new KlassDerived("me oh my");
      KlassDerived kret = li_boost_intrusive_ptr.derivedsmartptrreftest(k);
      String val = kret.getValue();
      verifyValue("me oh my derivedsmartptrreftest-Derived", val);
      verifyIntrusiveCount(2, k);
      verifyCount(2, k); // includes extra reference for upcast
      verifyIntrusiveCount(2, kret);
      verifyCount(2, kret);
    }
    // derived pass by shared_ptr pointer ref
    {
      KlassDerived k = new KlassDerived("me oh my");
      KlassDerived kret = li_boost_intrusive_ptr.derivedsmartptrpointerreftest(k);
      String val = kret.getValue();
      verifyValue("me oh my derivedsmartptrpointerreftest-Derived", val);
      verifyIntrusiveCount(2, k);
      verifyCount(2, k); // includes extra reference for upcast
      verifyIntrusiveCount(2, kret);
      verifyCount(2, kret);
    }
    // derived pass by pointer
    {
      KlassDerived k = new KlassDerived("me oh my");
      verifyCount(2, k); // includes an extra reference for the upcast in the proxy class
      KlassDerived kret = li_boost_intrusive_ptr.derivedpointertest(k);
      verifyCount(2, kret);
      String val = kret.getValue();
      verifyValue("me oh my derivedpointertest-Derived", val);
      verifyIntrusiveCount(1, k); //one shared_ptr has a null deleter
      verifyCount(2, k); // includes extra reference for upcast
      verifyIntrusiveCount(1, kret); //one shared_ptr has a null deleter
      verifyCount(2, kret);
    }
    // derived pass by ref
    {
      KlassDerived k = new KlassDerived("me oh my");
      verifyCount(2, k); // includes an extra reference for the upcast in the proxy class
      KlassDerived kret = li_boost_intrusive_ptr.derivedreftest(k);
      verifyCount(2, kret);
      String val = kret.getValue();
      verifyValue("me oh my derivedreftest-Derived", val);
      verifyIntrusiveCount(1, k); //one shared_ptr has a null deleter
      verifyCount(2, k); // includes extra reference for upcast
      verifyIntrusiveCount(1, kret); //one shared_ptr has a null deleter
      verifyCount(2, kret);
    }

    ////////////////////////////////// Derived and base class mixed ////////////////////////////////////////
    // pass by shared_ptr (mixed)
    {
      KlassDerived k = new KlassDerivedDerived("me oh my");
      KlassDerived kret = li_boost_intrusive_ptr.derivedsmartptrtest(k);
      String val = kret.getValue();
      verifyValue("me oh my derivedsmartptrtest-DerivedDerived", val);
      verifyIntrusiveCount(2, k);
      verifyCount(3, k); // an extra reference for the upcast in the proxy class
      verifyIntrusiveCount(2, kret);
      verifyCount(2, kret);
    }

    // pass by shared_ptr pointer (mixed)
    {
      KlassDerived k = new KlassDerivedDerived("me oh my");
      KlassDerived kret = li_boost_intrusive_ptr.derivedsmartptrpointertest(k);
      String val = kret.getValue();
      verifyValue("me oh my derivedsmartptrpointertest-DerivedDerived", val);
      verifyIntrusiveCount(2, k);
      verifyCount(3, k); // an extra reference for the upcast in the proxy class
      verifyIntrusiveCount(2, kret);
      verifyCount(2, kret);
    }

    // pass by shared_ptr reference (mixed)
    {
      KlassDerived k = new KlassDerivedDerived("me oh my");
      KlassDerived kret = li_boost_intrusive_ptr.derivedsmartptrreftest(k);
      String val = kret.getValue();
      verifyValue("me oh my derivedsmartptrreftest-DerivedDerived", val);
      verifyIntrusiveCount(2, k);
      verifyCount(3, k); // an extra reference for the upcast in the proxy class
      verifyIntrusiveCount(2, kret);
      verifyCount(2, kret);
    }

    // pass by shared_ptr pointer reference (mixed)
    {
      KlassDerived k = new KlassDerivedDerived("me oh my");
      KlassDerived kret = li_boost_intrusive_ptr.derivedsmartptrpointerreftest(k);
      String val = kret.getValue();
      verifyValue("me oh my derivedsmartptrpointerreftest-DerivedDerived", val);
      verifyIntrusiveCount(2, k);
      verifyCount(3, k); // an extra reference for the upcast in the proxy class
      verifyIntrusiveCount(2, kret);
      verifyCount(2, kret);
    }

    // pass by value (mixed)
    {
      KlassDerived k = new KlassDerivedDerived("me oh my");
      KlassDerived kret = li_boost_intrusive_ptr.derivedvaluetest(k);
      String val = kret.getValue();
      verifyValue("me oh my derivedvaluetest-Derived", val); // note slicing
      verifyIntrusiveCount(1, k);
      verifyCount(3, k); // an extra reference for the upcast in the proxy class
      verifyIntrusiveCount(1, kret);
      verifyCount(2, kret);
    }

    // pass by pointer (mixed)
    {
      KlassDerived k = new KlassDerivedDerived("me oh my");
      KlassDerived kret = li_boost_intrusive_ptr.derivedpointertest(k);
      String val = kret.getValue();
      verifyValue("me oh my derivedpointertest-DerivedDerived", val);
      verifyIntrusiveCount(1, k); //one shared_ptr has a null deleter
      verifyCount(3, k); // an extra reference for the upcast in the proxy class
      verifyIntrusiveCount(1, kret); //one shared_ptr has a null deleter
      verifyCount(2, kret);
    }

    // pass by ref (mixed)
    {
      KlassDerived k = new KlassDerivedDerived("me oh my");
      KlassDerived kret = li_boost_intrusive_ptr.derivedreftest(k);
      String val = kret.getValue();
      verifyValue("me oh my derivedreftest-DerivedDerived", val);
      verifyIntrusiveCount(1, k); //one shared_ptr has a null deleter
      verifyCount(3, k); // an extra reference for the upcast in the proxy class
      verifyIntrusiveCount(1, kret); //one shared_ptr has a null deleter
      verifyCount(2, kret);
    }

    ////////////////////////////////// Member variables ////////////////////////////////////////
    // smart pointer by value
    {
      MemberVariables m = new MemberVariables();
      Klass k = new Klass("smart member value");
      m.setSmartMemberValue(k);
      String val = k.getValue();
      verifyValue("smart member value", val);
      verifyIntrusiveCount(2, k);
      verifyCount(1, k);

      Klass kmember = m.getSmartMemberValue();
      val = kmember.getValue();
      verifyValue("smart member value", val);
      verifyIntrusiveCount(3, kmember);
      verifyIntrusiveCount(3, k);
      verifyCount(1, k);
      verifyCount(1, kmember);

      m.delete();
      verifyIntrusiveCount(2, kmember);
      verifyIntrusiveCount(2, k);
    }

    // smart pointer by pointer
    {
      MemberVariables m = new MemberVariables();
      Klass k = new Klass("smart member pointer");
      m.setSmartMemberPointer(k);
      String val = k.getValue();
      verifyValue("smart member pointer", val);
      verifyCount(1, k);
      verifyIntrusiveCount(2, k);

      Klass kmember = m.getSmartMemberPointer();
      val = kmember.getValue();
      verifyValue("smart member pointer", val);
      verifyIntrusiveCount(3, kmember);
      verifyCount(1, kmember);
      verifyIntrusiveCount(3, k);
      verifyCount(1, k);

      m.delete();
      verifyIntrusiveCount(2, kmember);
      verifyCount(1, kmember);
      verifyIntrusiveCount(2, k);
      verifyCount(1, k);
    }
    // smart pointer by reference
    {
      MemberVariables m = new MemberVariables();
      Klass k = new Klass("smart member reference");
      m.setSmartMemberReference(k);
      String val = k.getValue();
      verifyValue("smart member reference", val);
      verifyIntrusiveCount(2, k);
      verifyCount(1, k);

      Klass kmember = m.getSmartMemberReference();
      val = kmember.getValue();
      verifyValue("smart member reference", val);
      verifyIntrusiveCount(3, kmember);
      verifyCount(1, kmember);
      verifyIntrusiveCount(3, k);
      verifyCount(1, k);

      // The C++ reference refers to SmartMemberValue...
      m.setSmartMemberValue(k);
      Klass kmemberVal = m.getSmartMemberValue();
      val = kmember.getValue();
      verifyValue("smart member reference", val);
      verifyIntrusiveCount(5, kmemberVal);
      verifyCount(1, kmemberVal);
      verifyIntrusiveCount(5, kmember);
      verifyCount(1, kmember);
      verifyIntrusiveCount(5, k);
      verifyCount(1, k);

      m.delete();
      verifyIntrusiveCount(3, kmemberVal);
      verifyCount(1, kmemberVal);
      verifyIntrusiveCount(3, kmember);
      verifyCount(1, kmember);
      verifyIntrusiveCount(3, k);
      verifyCount(1, k);
    }

    //plain by value
    {
      MemberVariables m = new MemberVariables();
      Klass k = new Klass("plain member value");
      m.setMemberValue(k);
      String val = k.getValue();
      verifyValue("plain member value", val);
      verifyCount(1, k);

      Klass kmember = m.getMemberValue();
      val = kmember.getValue();
      verifyValue("plain member value", val);
      verifyCount(1, kmember);
      verifyCount(1, k);

      m.delete();
      verifyCount(1, kmember);
      verifyCount(1, k);
    }
    //plain by pointer
    {
      MemberVariables m = new MemberVariables();
      Klass k = new Klass("plain member pointer");
      m.setMemberPointer(k);
      String val = k.getValue();
      verifyValue("plain member pointer", val);
      verifyCount(1, k);

      Klass kmember = m.getMemberPointer();
      val = kmember.getValue();
      verifyValue("plain member pointer", val);
      verifyCount(1, kmember);
      verifyCount(1, k);

      m.delete();
      verifyCount(1, kmember);
      verifyCount(1, k);
    }
    //plain by reference
    {
      MemberVariables m = new MemberVariables();
      Klass k = new Klass("plain member reference");
      m.setMemberReference(k);
      String val = k.getValue();
      verifyValue("plain member reference", val);
      verifyCount(1, k);

      Klass kmember = m.getMemberReference();
      val = kmember.getValue();
      verifyValue("plain member reference", val);
      verifyCount(1, kmember);
      verifyCount(1, k);

      m.delete();
      verifyCount(1, kmember);
      verifyCount(1, k);
    }
    //null member variables
    {
      MemberVariables m = new MemberVariables();

      // shared_ptr by value
      Klass k = m.getSmartMemberValue();
      if (k != null)
        throw new RuntimeException("expected null");
      m.setSmartMemberValue(null);
      k = m.getSmartMemberValue();
      if (k != null)
        throw new RuntimeException("expected null");
      verifyCount(0, k);

      // plain by value
      try { m.setMemberValue(null); throw new RuntimeException("Failed to catch null pointer"); } catch (NullPointerException e) {}
    }
}
private void toIgnore() {
    ////////////////////////////////// Global variables ////////////////////////////////////////
    // smart pointer
    {
      Klass kglobal = li_boost_intrusive_ptr.getGlobalSmartValue();
      if (kglobal != null)
        throw new RuntimeException("expected null");

      Klass k = new Klass("smart global value");
      li_boost_intrusive_ptr.setGlobalSmartValue(k);
      verifyIntrusiveCount(2, k);
      verifyCount(1, k);

      kglobal = li_boost_intrusive_ptr.getGlobalSmartValue();
      String val = kglobal.getValue();
      verifyValue("smart global value", val);
      verifyIntrusiveCount(3, kglobal);
      verifyCount(1, kglobal);
      verifyIntrusiveCount(3, k);
      verifyCount(1, k);
      verifyValue("smart global value", li_boost_intrusive_ptr.getGlobalSmartValue().getValue());
      li_boost_intrusive_ptr.setGlobalSmartValue(null);
    }
    // plain value
    {
      Klass kglobal;

      Klass k = new Klass("global value");
      li_boost_intrusive_ptr.setGlobalValue(k);
      verifyCount(1, k);

      kglobal = li_boost_intrusive_ptr.getGlobalValue();
      String val = kglobal.getValue();
      verifyValue("global value", val);
      verifyCount(1, kglobal);
      verifyCount(1, k);
      verifyValue("global value", li_boost_intrusive_ptr.getGlobalValue().getValue());

      try { li_boost_intrusive_ptr.setGlobalValue(null); throw new RuntimeException("Failed to catch null pointer"); } catch (NullPointerException e) {}
    }
    //plain pointer
    {
      Klass kglobal = li_boost_intrusive_ptr.getGlobalPointer();
      if (kglobal != null)
        throw new RuntimeException("expected null");

      Klass k = new Klass("global pointer");
      li_boost_intrusive_ptr.setGlobalPointer(k);
      verifyCount(1, k);

      kglobal = li_boost_intrusive_ptr.getGlobalPointer();
      String val = kglobal.getValue();
      verifyValue("global pointer", val);
      verifyCount(1, kglobal);
      verifyCount(1, k);
      li_boost_intrusive_ptr.setGlobalPointer(null);
    }

    // plain reference
    {
      Klass kglobal;

      Klass k = new Klass("global reference");
      li_boost_intrusive_ptr.setGlobalReference(k);
      verifyCount(1, k);

      kglobal = li_boost_intrusive_ptr.getGlobalReference();
      String val = kglobal.getValue();
      verifyValue("global reference", val);
      verifyCount(1, kglobal);
      verifyCount(1, k);

      try { li_boost_intrusive_ptr.setGlobalReference(null); throw new RuntimeException("Failed to catch null pointer"); } catch (NullPointerException e) {}
    }

    ////////////////////////////////// Templates ////////////////////////////////////////
    {
      PairIntDouble pid = new PairIntDouble(10, 20.2);
      if (pid.getBaseVal1() != 20 || pid.getBaseVal2() != 40.4)
        throw new RuntimeException("Base values wrong");
      if (pid.getVal1() != 10 || pid.getVal2() != 20.2)
        throw new RuntimeException("Derived Values wrong");
    }
  }
  private void verifyValue(String expected, String got) {
    if (!expected.equals(got))
      throw new RuntimeException("verify value failed. Expected: " + expected + " Got: " + got + " loopCount: " + loopCount);
  }
  private void verifyCount(int expected, Klass k) {
    int got = li_boost_intrusive_ptr.use_count(k);
    if (expected != got)
      throw new RuntimeException("verify use_count failed. Expected: " + expected + " Got: " + got + " loopCount: " + loopCount);
  }
  private void verifyCount(int expected, KlassDerived kd) {
      int got = li_boost_intrusive_ptr.use_count(kd);
      if (expected != got)
        throw new RuntimeException("verify use_count failed. Expected: " + expected + " Got: " + got + " loopCount: " + loopCount);
  }
  private void verifyCount(int expected, KlassDerivedDerived kdd) {
      int got = li_boost_intrusive_ptr.use_count(kdd);
      if (expected != got)
        throw new RuntimeException("verify use_count failed. Expected: " + expected + " Got: " + got + " loopCount: " + loopCount);
  }
  private void verifyIntrusiveCount(int expected, Klass k) {
    int got = k.use_count();
    if (expected != got)
      throw new RuntimeException("verify use_count failed. Expected: " + expected + " Got: " + got + " loopCount: " + loopCount);
  }
  private void verifyIntrusiveCount(int expected, KlassDerived kd) {
      int got = kd.use_count();
      if (expected != got)
        throw new RuntimeException("verify use_count failed. Expected: " + expected + " Got: " + got + " loopCount: " + loopCount);
  }
  private void verifyIntrusiveCount(int expected, KlassDerivedDerived kdd) {
      int got = kdd.use_count();
      if (expected != got)
        throw new RuntimeException("verify use_count failed. Expected: " + expected + " Got: " + got + " loopCount: " + loopCount);
  }
}
