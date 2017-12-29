import global_namespace.*;

public class global_namespace_runme {

  static {
    try {
	System.loadLibrary("global_namespace");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) {

    Klass1 k1 = new Klass1();
    Klass2 k2 = new Klass2();
    Klass3 k3 = new Klass3();
    Klass4 k4 = new Klass4();
    Klass5 k5 = new Klass5();
    Klass6 k6 = new Klass6();
    Klass7 k7 = new Klass7();

    KlassMethods.methodA(k1, k2, k3, k4, k5, k6, k7);
    KlassMethods.methodB(k1, k2, k3, k4, k5, k6, k7);

    k1 = global_namespace.getKlass1A();
    k2 = global_namespace.getKlass2A();
    k3 = global_namespace.getKlass3A();
    k4 = global_namespace.getKlass4A();
    k5 = global_namespace.getKlass5A();
    k6 = global_namespace.getKlass6A();
    k7 = global_namespace.getKlass7A();

    KlassMethods.methodA(k1, k2, k3, k4, k5, k6, k7);
    KlassMethods.methodB(k1, k2, k3, k4, k5, k6, k7);

    k1 = global_namespace.getKlass1B();
    k2 = global_namespace.getKlass2B();
    k3 = global_namespace.getKlass3B();
    k4 = global_namespace.getKlass4B();
    k5 = global_namespace.getKlass5B();
    k6 = global_namespace.getKlass6B();
    k7 = global_namespace.getKlass7B();

    KlassMethods.methodA(k1, k2, k3, k4, k5, k6, k7);
    KlassMethods.methodB(k1, k2, k3, k4, k5, k6, k7);

    XYZMethods.methodA(new XYZ1(), new XYZ2(), new XYZ3(), new XYZ4(), new XYZ5(), new XYZ6(), new XYZ7());
    XYZMethods.methodB(new XYZ1(), new XYZ2(), new XYZ3(), new XYZ4(), new XYZ5(), new XYZ6(), new XYZ7());

    TheEnumMethods.methodA(TheEnum1.theenum1, TheEnum2.theenum2, TheEnum3.theenum3);
    TheEnumMethods.methodA(TheEnum1.theenum1, TheEnum2.theenum2, TheEnum3.theenum3);
  }
}
