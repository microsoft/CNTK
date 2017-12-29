import cpp11_strongly_typed_enumerations.*;

public class cpp11_strongly_typed_enumerations_runme {

  static {
    try {
        System.loadLibrary("cpp11_strongly_typed_enumerations");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static int enumCheck(int actual, int expected) {
    if (actual != expected)
      throw new RuntimeException("Enum value mismatch. Expected " + expected + " Actual: " + actual);
    return expected + 1;
  }

  public static void main(String argv[]) {
    int val = 0;
    val = enumCheck(Enum1.Val1.swigValue(), val);
    val = enumCheck(Enum1.Val2.swigValue(), val);
    val = enumCheck(Enum1.Val3.swigValue(), 13);
    val = enumCheck(Enum1.Val4.swigValue(), val);
    val = enumCheck(Enum1.Val5a.swigValue(), 13);
    val = enumCheck(Enum1.Val6a.swigValue(), val);

    val = 0;
    val = enumCheck(Enum2.Val1.swigValue(), val);
    val = enumCheck(Enum2.Val2.swigValue(), val);
    val = enumCheck(Enum2.Val3.swigValue(), 23);
    val = enumCheck(Enum2.Val4.swigValue(), val);
    val = enumCheck(Enum2.Val5b.swigValue(), 23);
    val = enumCheck(Enum2.Val6b.swigValue(), val);

    val = 0;
    val = enumCheck(Enum4.Val1.swigValue(), val);
    val = enumCheck(Enum4.Val2.swigValue(), val);
    val = enumCheck(Enum4.Val3.swigValue(), 43);
    val = enumCheck(Enum4.Val4.swigValue(), val);

    val = 0;
    val = enumCheck(Enum5.Val1.swigValue(), val);
    val = enumCheck(Enum5.Val2.swigValue(), val);
    val = enumCheck(Enum5.Val3.swigValue(), 53);
    val = enumCheck(Enum5.Val4.swigValue(), val);

    val = 0;
    val = enumCheck(Enum6.Val1.swigValue(), val);
    val = enumCheck(Enum6.Val2.swigValue(), val);
    val = enumCheck(Enum6.Val3.swigValue(), 63);
    val = enumCheck(Enum6.Val4.swigValue(), val);

    val = 0;
    val = enumCheck(Enum7td.Val1.swigValue(), val);
    val = enumCheck(Enum7td.Val2.swigValue(), val);
    val = enumCheck(Enum7td.Val3.swigValue(), 73);
    val = enumCheck(Enum7td.Val4.swigValue(), val);

    val = 0;
    val = enumCheck(Enum8.Val1.swigValue(), val);
    val = enumCheck(Enum8.Val2.swigValue(), val);
    val = enumCheck(Enum8.Val3.swigValue(), 83);
    val = enumCheck(Enum8.Val4.swigValue(), val);

    val = 0;
    val = enumCheck(Enum10.Val1.swigValue(), val);
    val = enumCheck(Enum10.Val2.swigValue(), val);
    val = enumCheck(Enum10.Val3.swigValue(), 103);
    val = enumCheck(Enum10.Val4.swigValue(), val);

    val = 0;
    val = enumCheck(Class1.Enum12.Val1.swigValue(), 1121);
    val = enumCheck(Class1.Enum12.Val2.swigValue(), 1122);
    val = enumCheck(Class1.Enum12.Val3.swigValue(), val);
    val = enumCheck(Class1.Enum12.Val4.swigValue(), val);
    val = enumCheck(Class1.Enum12.Val5c.swigValue(), 1121);
    val = enumCheck(Class1.Enum12.Val6c.swigValue(), val);

    val = 0;
    val = enumCheck(Class1.Enum13.Val1.swigValue(), 1131);
    val = enumCheck(Class1.Enum13.Val2.swigValue(), 1132);
    val = enumCheck(Class1.Enum13.Val3.swigValue(), val);
    val = enumCheck(Class1.Enum13.Val4.swigValue(), val);
    val = enumCheck(Class1.Enum13.Val5d.swigValue(), 1131);
    val = enumCheck(Class1.Enum13.Val6d.swigValue(), val);

    val = 0;
    val = enumCheck(Class1.Enum14.Val1.swigValue(), 1141);
    val = enumCheck(Class1.Enum14.Val2.swigValue(), 1142);
    val = enumCheck(Class1.Enum14.Val3.swigValue(), val);
    val = enumCheck(Class1.Enum14.Val4.swigValue(), val);
    val = enumCheck(Class1.Enum14.Val5e.swigValue(), 1141);
    val = enumCheck(Class1.Enum14.Val6e.swigValue(), val);

    val = 0;
    val = enumCheck(Class1.Struct1.Enum12.Val1.swigValue(), 3121);
    val = enumCheck(Class1.Struct1.Enum12.Val2.swigValue(), 3122);
    val = enumCheck(Class1.Struct1.Enum12.Val3.swigValue(), val);
    val = enumCheck(Class1.Struct1.Enum12.Val4.swigValue(), val);
    val = enumCheck(Class1.Struct1.Enum12.Val5f.swigValue(), 3121);
    val = enumCheck(Class1.Struct1.Enum12.Val6f.swigValue(), val);

    val = 0;
    val = enumCheck(Class1.Struct1.Enum13.Val1.swigValue(), 3131);
    val = enumCheck(Class1.Struct1.Enum13.Val2.swigValue(), 3132);
    val = enumCheck(Class1.Struct1.Enum13.Val3.swigValue(), val);
    val = enumCheck(Class1.Struct1.Enum13.Val4.swigValue(), val);

    val = 0;
    val = enumCheck(Class1.Struct1.Enum14.Val1.swigValue(), 3141);
    val = enumCheck(Class1.Struct1.Enum14.Val2.swigValue(), 3142);
    val = enumCheck(Class1.Struct1.Enum14.Val3.swigValue(), val);
    val = enumCheck(Class1.Struct1.Enum14.Val4.swigValue(), val);
    val = enumCheck(Class1.Struct1.Enum14.Val5g.swigValue(), 3141);
    val = enumCheck(Class1.Struct1.Enum14.Val6g.swigValue(), val);

    val = 0;
    val = enumCheck(Class2.Enum12.Val1.swigValue(), 2121);
    val = enumCheck(Class2.Enum12.Val2.swigValue(), 2122);
    val = enumCheck(Class2.Enum12.Val3.swigValue(), val);
    val = enumCheck(Class2.Enum12.Val4.swigValue(), val);
    val = enumCheck(Class2.Enum12.Val5h.swigValue(), 2121);
    val = enumCheck(Class2.Enum12.Val6h.swigValue(), val);

    val = 0;
    val = enumCheck(Class2.Enum13.Val1.swigValue(), 2131);
    val = enumCheck(Class2.Enum13.Val2.swigValue(), 2132);
    val = enumCheck(Class2.Enum13.Val3.swigValue(), val);
    val = enumCheck(Class2.Enum13.Val4.swigValue(), val);
    val = enumCheck(Class2.Enum13.Val5i.swigValue(), 2131);
    val = enumCheck(Class2.Enum13.Val6i.swigValue(), val);

    val = 0;
    val = enumCheck(Class2.Enum14.Val1.swigValue(), 2141);
    val = enumCheck(Class2.Enum14.Val2.swigValue(), 2142);
    val = enumCheck(Class2.Enum14.Val3.swigValue(), val);
    val = enumCheck(Class2.Enum14.Val4.swigValue(), val);
    val = enumCheck(Class2.Enum14.Val5j.swigValue(), 2141);
    val = enumCheck(Class2.Enum14.Val6j.swigValue(), val);

    val = 0;
    val = enumCheck(Class2.Struct1.Enum12.Val1.swigValue(), 4121);
    val = enumCheck(Class2.Struct1.Enum12.Val2.swigValue(), 4122);
    val = enumCheck(Class2.Struct1.Enum12.Val3.swigValue(), val);
    val = enumCheck(Class2.Struct1.Enum12.Val4.swigValue(), val);
    val = enumCheck(Class2.Struct1.Enum12.Val5k.swigValue(), 4121);
    val = enumCheck(Class2.Struct1.Enum12.Val6k.swigValue(), val);

    val = 0;
    val = enumCheck(Class2.Struct1.Enum13.Val1.swigValue(), 4131);
    val = enumCheck(Class2.Struct1.Enum13.Val2.swigValue(), 4132);
    val = enumCheck(Class2.Struct1.Enum13.Val3.swigValue(), val);
    val = enumCheck(Class2.Struct1.Enum13.Val4.swigValue(), val);
    val = enumCheck(Class2.Struct1.Enum13.Val5l.swigValue(), 4131);
    val = enumCheck(Class2.Struct1.Enum13.Val6l.swigValue(), val);

    val = 0;
    val = enumCheck(Class2.Struct1.Enum14.Val1.swigValue(), 4141);
    val = enumCheck(Class2.Struct1.Enum14.Val2.swigValue(), 4142);
    val = enumCheck(Class2.Struct1.Enum14.Val3.swigValue(), val);
    val = enumCheck(Class2.Struct1.Enum14.Val4.swigValue(), val);
    val = enumCheck(Class2.Struct1.Enum14.Val5m.swigValue(), 4141);
    val = enumCheck(Class2.Struct1.Enum14.Val6m.swigValue(), val);

    Class1 class1 = new Class1();
    enumCheck(class1.class1Test1(Enum1.Val5a).swigValue(), 13);
    enumCheck(class1.class1Test2(Class1.Enum12.Val5c).swigValue(), 1121);
    enumCheck(class1.class1Test3(Class1.Struct1.Enum12.Val5f).swigValue(), 3121);

    enumCheck(cpp11_strongly_typed_enumerations.globalTest1(Enum1.Val5a).swigValue(), 13);
    enumCheck(cpp11_strongly_typed_enumerations.globalTest2(Class1.Enum12.Val5c).swigValue(), 1121);
    enumCheck(cpp11_strongly_typed_enumerations.globalTest3(Class1.Struct1.Enum12.Val5f).swigValue(), 3121);
  }
}
