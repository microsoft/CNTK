
import extend_default.Override;
import extend_default.*;

public class extend_default_runme {
  static {
    try {
        System.loadLibrary("extend_default");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) 
  {
    // %extend before the class definition
    {
      Before ex = new Before();
      if (ex.getI() != -1.0 && ex.getD() != -1.0)
        throw new RuntimeException("Before constructor 1 failed");

      ex = new Before(10);
      if (ex.getI() != 10.0 && ex.getD() != -1.0)
        throw new RuntimeException("Before constructor 2 failed");

      ex = new Before(20, 30.0);
      if (ex.getI() != 20 && ex.getD() != 30.0)
        throw new RuntimeException("Before constructor 3 failed");
    }
    {
      Before ex = new Before();
      if (ex.AddedMethod() != -2.0)
        throw new RuntimeException("Before AddedMethod 1 failed");
      if (ex.AddedMethod(-2) != -3.0)
        throw new RuntimeException("Before AddedMethod 2 failed");
      if (ex.AddedMethod(-10, -10.0) != -20)
        throw new RuntimeException("Before AddedMethod 3 failed");
    }
    {
      if (Before.AddedStaticMethod() != -2.0)
        throw new RuntimeException("Before AddedStaticMethod 1 failed");
      if (Before.AddedStaticMethod(-2) != -3.0)
        throw new RuntimeException("Before AddedStaticMethod 2 failed");
      if (Before.AddedStaticMethod(-10, -10.0) != -20)
        throw new RuntimeException("Before AddedStaticMethod 3 failed");
    }

    // %extend after the class definition
    {
      After ex = new After();
      if (ex.getI() != -1.0 && ex.getD() != -1.0)
        throw new RuntimeException("After constructor 1 failed");

      ex = new After(10);
      if (ex.getI() != 10.0 && ex.getD() != -1.0)
        throw new RuntimeException("After constructor 2 failed");

      ex = new After(20, 30.0);
      if (ex.getI() != 20 && ex.getD() != 30.0)
        throw new RuntimeException("After constructor 3 failed");
    }
    {
      After ex = new After();
      if (ex.AddedMethod() != -2.0)
        throw new RuntimeException("After AddedMethod 1 failed");
      if (ex.AddedMethod(-2) != -3.0)
        throw new RuntimeException("After AddedMethod 2 failed");
      if (ex.AddedMethod(-10, -10.0) != -20)
        throw new RuntimeException("After AddedMethod 3 failed");
    }
    {
      if (After.AddedStaticMethod() != -2.0)
        throw new RuntimeException("After AddedStaticMethod 1 failed");
      if (After.AddedStaticMethod(-2) != -3.0)
        throw new RuntimeException("After AddedStaticMethod 2 failed");
      if (After.AddedStaticMethod(-10, -10.0) != -20)
        throw new RuntimeException("After AddedStaticMethod 3 failed");
    }

    // %extend before the class definition - with overloading and default args
    {
      OverBefore ex = new OverBefore();
      if (ex.getI() != -1.0 && ex.getD() != -1.0)
        throw new RuntimeException("OverBefore constructor 1 failed");

      ex = new OverBefore(10);
      if (ex.getI() != 10.0 && ex.getD() != -1.0)
        throw new RuntimeException("OverBefore constructor 2 failed");

      ex = new OverBefore(20, 30.0);
      if (ex.getI() != 20 && ex.getD() != 30.0)
        throw new RuntimeException("OverBefore constructor 3 failed");
    }
    {
      OverBefore ex = new OverBefore();
      if (ex.AddedMethod() != -2.0)
        throw new RuntimeException("OverBefore AddedMethod 1 failed");
      if (ex.AddedMethod(-2) != -3.0)
        throw new RuntimeException("OverBefore AddedMethod 2 failed");
      if (ex.AddedMethod(-10, -10.0) != -20)
        throw new RuntimeException("OverBefore AddedMethod 3 failed");
    }
    {
      if (OverBefore.AddedStaticMethod() != -2.0)
        throw new RuntimeException("OverBefore AddedStaticMethod 1 failed");
      if (OverBefore.AddedStaticMethod(-2) != -3.0)
        throw new RuntimeException("OverBefore AddedStaticMethod 2 failed");
      if (OverBefore.AddedStaticMethod(-10, -10.0) != -20)
        throw new RuntimeException("OverBefore AddedStaticMethod 3 failed");
    }
    {
      OverBefore ex = new OverBefore("hello");
      if (ex.getI() != -2.0 && ex.getD() != -2.0)
        throw new RuntimeException("OverBefore overload constructor 1 failed");

      ex = new OverBefore("hello", 10);
      if (ex.getI() != 10.0 && ex.getD() != -1.0)
        throw new RuntimeException("OverBefore overload constructor 2 failed");

      ex = new OverBefore("hello", 20, 30.0);
      if (ex.getI() != 20 && ex.getD() != 30.0)
        throw new RuntimeException("OverBefore overload constructor 3 failed");
    }
    {
      OverBefore ex = new OverBefore("hello");
      if (ex.AddedMethod("hello") != -2.0)
        throw new RuntimeException("OverBefore overload AddedMethod 1 failed");
      if (ex.AddedMethod("hello", -2) != -3.0)
        throw new RuntimeException("OverBefore overload AddedMethod 2 failed");
      if (ex.AddedMethod("hello", -10, -10.0) != -20)
        throw new RuntimeException("OverBefore overload AddedMethod 3 failed");
    }
    {
      if (OverBefore.AddedStaticMethod("hello") != -2.0)
        throw new RuntimeException("OverBefore overload AddedStaticMethod 1 failed");
      if (OverBefore.AddedStaticMethod("hello", -2) != -3.0)
        throw new RuntimeException("OverBefore overload AddedStaticMethod 2 failed");
      if (OverBefore.AddedStaticMethod("hello", -10, -10.0) != -20)
        throw new RuntimeException("OverBefore overload AddedStaticMethod 3 failed");
    }

    // %extend after the class definition - with overloading and default args
    {
      OverAfter ex = new OverAfter();
      if (ex.getI() != -1.0 && ex.getD() != -1.0)
        throw new RuntimeException("OverAfter constructor 1 failed");

      ex = new OverAfter(10);
      if (ex.getI() != 10.0 && ex.getD() != -1.0)
        throw new RuntimeException("OverAfter constructor 2 failed");

      ex = new OverAfter(20, 30.0);
      if (ex.getI() != 20 && ex.getD() != 30.0)
        throw new RuntimeException("OverAfter constructor 3 failed");
    }
    {
      OverAfter ex = new OverAfter();
      if (ex.AddedMethod() != -2.0)
        throw new RuntimeException("OverAfter AddedMethod 1 failed");
      if (ex.AddedMethod(-2) != -3.0)
        throw new RuntimeException("OverAfter AddedMethod 2 failed");
      if (ex.AddedMethod(-10, -10.0) != -20)
        throw new RuntimeException("OverAfter AddedMethod 3 failed");
    }
    {
      if (OverAfter.AddedStaticMethod() != -2.0)
        throw new RuntimeException("OverAfter AddedStaticMethod 1 failed");
      if (OverAfter.AddedStaticMethod(-2) != -3.0)
        throw new RuntimeException("OverAfter AddedStaticMethod 2 failed");
      if (OverAfter.AddedStaticMethod(-10, -10.0) != -20)
        throw new RuntimeException("OverAfter AddedStaticMethod 3 failed");
    }
    {
      OverAfter ex = new OverAfter("hello");
      if (ex.getI() != -2.0 && ex.getD() != -2.0)
        throw new RuntimeException("OverAfter overload constructor 1 failed");

      ex = new OverAfter("hello", 10);
      if (ex.getI() != 10.0 && ex.getD() != -1.0)
        throw new RuntimeException("OverAfter overload constructor 2 failed");

      ex = new OverAfter("hello", 20, 30.0);
      if (ex.getI() != 20 && ex.getD() != 30.0)
        throw new RuntimeException("OverAfter overload constructor 3 failed");
    }
    {
      OverAfter ex = new OverAfter("hello");
      if (ex.AddedMethod("hello") != -2.0)
        throw new RuntimeException("OverAfter overload AddedMethod 1 failed");
      if (ex.AddedMethod("hello", -2) != -3.0)
        throw new RuntimeException("OverAfter overload AddedMethod 2 failed");
      if (ex.AddedMethod("hello", -10, -10.0) != -20)
        throw new RuntimeException("OverAfter overload AddedMethod 3 failed");
    }
    {
      if (OverAfter.AddedStaticMethod("hello") != -2.0)
        throw new RuntimeException("OverAfter overload AddedStaticMethod 1 failed");
      if (OverAfter.AddedStaticMethod("hello", -2) != -3.0)
        throw new RuntimeException("OverAfter overload AddedStaticMethod 2 failed");
      if (OverAfter.AddedStaticMethod("hello", -10, -10.0) != -20)
        throw new RuntimeException("OverAfter overload AddedStaticMethod 3 failed");
    }

    // Override
    {
      Override o = new Override();

      if (o.over() != -1)
        throw new RuntimeException("override test 1 failed");
      if (o.over(10) != 10*10)
        throw new RuntimeException("override test 2 failed");

      if (o.ride() != -1)
        throw new RuntimeException("override test 3 failed");
      if (o.ride(10) != 10)
        throw new RuntimeException("override test 4 failed");

      if (o.overload() != -10)
        throw new RuntimeException("override test 5 failed");
      if (o.overload(10) != 10*10)
        throw new RuntimeException("override test 6 failed");
    }
  }
}

