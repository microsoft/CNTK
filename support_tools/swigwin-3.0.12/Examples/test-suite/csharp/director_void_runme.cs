using System;
using director_voidNamespace;

public class runme
{
  private static void WaitForGC()
  {
    System.GC.Collect();
    System.GC.WaitForPendingFinalizers();
    System.Threading.Thread.Sleep(10);
  }

  static void Main()
  {
    runme r = new runme();
    r.run();
  }

  void run()
  {
    Caller caller = new Caller();
    {
      DirectorVoidPointer dvp = new DirectorVoidPointer(5);
      int x = caller.callVirtualIn(dvp, 6);
      if (x != 106)
        throw new Exception("Fail1 should be 106, got " + x);
      global::System.IntPtr ptr = dvp.nonVirtualVoidPtrOut();
      x = Caller.VoidToInt(ptr);
      if (x != 106)
        throw new Exception("Fail2 should be 106, got " + x);
      x = Caller.VoidToInt(dvp.voidPtrOut());
      if (x != 106)
        throw new Exception("Fail3 should be 106, got " + x);
    }

    {
      DirectorVoidPointer dvp = new director_void_VoidPointer(5);
      int x = caller.callVirtualIn(dvp, 6);
      if (x != 12)
        throw new Exception("Fail1 should be 12, got " + x);
      global::System.IntPtr ptr = dvp.nonVirtualVoidPtrOut();
      x = Caller.VoidToInt(ptr);
      if (x != 25)
        throw new Exception("Fail2 should be 25, got " + x);
      x = Caller.VoidToInt(dvp.voidPtrOut());
      if (x != 1234)
        throw new Exception("Fail3 should be 1234, got " + x);
    }

    {
      DirectorVoidPointer dvp = new DirectorVoidPointer(10);
      int x = caller.callVirtualOut(dvp);
      if (x != 10)
        throw new Exception("Bad1 should be 10, got " + x);
      global::System.IntPtr ptr = dvp.nonVirtualVoidPtrOut();
      x = dvp.nonVirtualVoidPtrIn(ptr);
      if (x != 110)
        throw new Exception("Bad2 should be 110, got " + x);
    }
    {
      DirectorVoidPointer dvp = new director_void_VoidPointer(10);
      int x = caller.callVirtualOut(dvp);
      if (x != 1234)
        throw new Exception("Bad3 should be 1234, got " + x);
      global::System.IntPtr ptr = dvp.nonVirtualVoidPtrOut();
      x = dvp.nonVirtualVoidPtrIn(ptr);
      if (x != 1334)
        throw new Exception("Bad4 should be 1334, got " + x);
    }
  }
}

class director_void_VoidPointer : DirectorVoidPointer {
  public director_void_VoidPointer(int num) : base(num*num) {
  }
  public override int voidPtrIn(global::System.IntPtr p) {
    return Caller.VoidToInt(p) * 2;
  }
  public override global::System.IntPtr voidPtrOut() {
    setNewValue(1234);
    return nonVirtualVoidPtrOut();
  }
}
