
import multiple_inheritance.*;

public class multiple_inheritance_runme {

  static {
    try {
	System.loadLibrary("multiple_inheritance");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) {
    FooBar fooBar = new FooBar();
    fooBar.foo();

    IgnoreDerived1 ignoreDerived1 = new IgnoreDerived1();
    ignoreDerived1.bar();

    IgnoreDerived2 ignoreDerived2 = new IgnoreDerived2();
    ignoreDerived2.bar();

    IgnoreDerived3 ignoreDerived3 = new IgnoreDerived3();
    ignoreDerived3.bar();

    IgnoreDerived4 ignoreDerived4 = new IgnoreDerived4();
    ignoreDerived4.bar();
  }
}

