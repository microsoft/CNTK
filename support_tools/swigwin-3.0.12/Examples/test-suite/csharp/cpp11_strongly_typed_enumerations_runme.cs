using System;
using cpp11_strongly_typed_enumerationsNamespace;

public class cpp11_strongly_typed_enumerations_runme {

  public static int enumCheck(int actual, int expected) {
    if (actual != expected)
      throw new ApplicationException("Enum value mismatch. Expected " + expected + " Actual: " + actual);
    return expected + 1;
  }

  public static void Main() {
    int val = 0;
    val = enumCheck((int)Enum1.Val1, val);
    val = enumCheck((int)Enum1.Val2, val);
    val = enumCheck((int)Enum1.Val3, 13);
    val = enumCheck((int)Enum1.Val4, val);
    val = enumCheck((int)Enum1.Val5a, 13);
    val = enumCheck((int)Enum1.Val6a, val);

    val = 0;
    val = enumCheck((int)Enum2.Val1, val);
    val = enumCheck((int)Enum2.Val2, val);
    val = enumCheck((int)Enum2.Val3, 23);
    val = enumCheck((int)Enum2.Val4, val);
    val = enumCheck((int)Enum2.Val5b, 23);
    val = enumCheck((int)Enum2.Val6b, val);

    val = 0;
    val = enumCheck((int)Enum4.Val1, val);
    val = enumCheck((int)Enum4.Val2, val);
    val = enumCheck((int)Enum4.Val3, 43);
    val = enumCheck((int)Enum4.Val4, val);

    val = 0;
    val = enumCheck((int)Enum5.Val1, val);
    val = enumCheck((int)Enum5.Val2, val);
    val = enumCheck((int)Enum5.Val3, 53);
    val = enumCheck((int)Enum5.Val4, val);

    val = 0;
    val = enumCheck((int)Enum6.Val1, val);
    val = enumCheck((int)Enum6.Val2, val);
    val = enumCheck((int)Enum6.Val3, 63);
    val = enumCheck((int)Enum6.Val4, val);

    val = 0;
    val = enumCheck((int)Enum7td.Val1, val);
    val = enumCheck((int)Enum7td.Val2, val);
    val = enumCheck((int)Enum7td.Val3, 73);
    val = enumCheck((int)Enum7td.Val4, val);

    val = 0;
    val = enumCheck((int)Enum8.Val1, val);
    val = enumCheck((int)Enum8.Val2, val);
    val = enumCheck((int)Enum8.Val3, 83);
    val = enumCheck((int)Enum8.Val4, val);

    val = 0;
    val = enumCheck((int)Enum10.Val1, val);
    val = enumCheck((int)Enum10.Val2, val);
    val = enumCheck((int)Enum10.Val3, 103);
    val = enumCheck((int)Enum10.Val4, val);

    val = 0;
    val = enumCheck((int)Class1.Enum12.Val1, 1121);
    val = enumCheck((int)Class1.Enum12.Val2, 1122);
    val = enumCheck((int)Class1.Enum12.Val3, val);
    val = enumCheck((int)Class1.Enum12.Val4, val);
    val = enumCheck((int)Class1.Enum12.Val5c, 1121);
    val = enumCheck((int)Class1.Enum12.Val6c, val);

    val = 0;
    val = enumCheck((int)Class1.Enum13.Val1, 1131);
    val = enumCheck((int)Class1.Enum13.Val2, 1132);
    val = enumCheck((int)Class1.Enum13.Val3, val);
    val = enumCheck((int)Class1.Enum13.Val4, val);
    val = enumCheck((int)Class1.Enum13.Val5d, 1131);
    val = enumCheck((int)Class1.Enum13.Val6d, val);

    val = 0;
    val = enumCheck((int)Class1.Enum14.Val1, 1141);
    val = enumCheck((int)Class1.Enum14.Val2, 1142);
    val = enumCheck((int)Class1.Enum14.Val3, val);
    val = enumCheck((int)Class1.Enum14.Val4, val);
    val = enumCheck((int)Class1.Enum14.Val5e, 1141);
    val = enumCheck((int)Class1.Enum14.Val6e, val);

    val = 0;
    val = enumCheck((int)Class1.Struct1.Enum12.Val1, 3121);
    val = enumCheck((int)Class1.Struct1.Enum12.Val2, 3122);
    val = enumCheck((int)Class1.Struct1.Enum12.Val3, val);
    val = enumCheck((int)Class1.Struct1.Enum12.Val4, val);
    val = enumCheck((int)Class1.Struct1.Enum12.Val5f, 3121);
    val = enumCheck((int)Class1.Struct1.Enum12.Val6f, val);

    val = 0;
    val = enumCheck((int)Class1.Struct1.Enum13.Val1, 3131);
    val = enumCheck((int)Class1.Struct1.Enum13.Val2, 3132);
    val = enumCheck((int)Class1.Struct1.Enum13.Val3, val);
    val = enumCheck((int)Class1.Struct1.Enum13.Val4, val);

    val = 0;
    val = enumCheck((int)Class1.Struct1.Enum14.Val1, 3141);
    val = enumCheck((int)Class1.Struct1.Enum14.Val2, 3142);
    val = enumCheck((int)Class1.Struct1.Enum14.Val3, val);
    val = enumCheck((int)Class1.Struct1.Enum14.Val4, val);
    val = enumCheck((int)Class1.Struct1.Enum14.Val5g, 3141);
    val = enumCheck((int)Class1.Struct1.Enum14.Val6g, val);

    val = 0;
    val = enumCheck((int)Class2.Enum12.Val1, 2121);
    val = enumCheck((int)Class2.Enum12.Val2, 2122);
    val = enumCheck((int)Class2.Enum12.Val3, val);
    val = enumCheck((int)Class2.Enum12.Val4, val);
    val = enumCheck((int)Class2.Enum12.Val5h, 2121);
    val = enumCheck((int)Class2.Enum12.Val6h, val);

    val = 0;
    val = enumCheck((int)Class2.Enum13.Val1, 2131);
    val = enumCheck((int)Class2.Enum13.Val2, 2132);
    val = enumCheck((int)Class2.Enum13.Val3, val);
    val = enumCheck((int)Class2.Enum13.Val4, val);
    val = enumCheck((int)Class2.Enum13.Val5i, 2131);
    val = enumCheck((int)Class2.Enum13.Val6i, val);

    val = 0;
    val = enumCheck((int)Class2.Enum14.Val1, 2141);
    val = enumCheck((int)Class2.Enum14.Val2, 2142);
    val = enumCheck((int)Class2.Enum14.Val3, val);
    val = enumCheck((int)Class2.Enum14.Val4, val);
    val = enumCheck((int)Class2.Enum14.Val5j, 2141);
    val = enumCheck((int)Class2.Enum14.Val6j, val);

    val = 0;
    val = enumCheck((int)Class2.Struct1.Enum12.Val1, 4121);
    val = enumCheck((int)Class2.Struct1.Enum12.Val2, 4122);
    val = enumCheck((int)Class2.Struct1.Enum12.Val3, val);
    val = enumCheck((int)Class2.Struct1.Enum12.Val4, val);
    val = enumCheck((int)Class2.Struct1.Enum12.Val5k, 4121);
    val = enumCheck((int)Class2.Struct1.Enum12.Val6k, val);

    val = 0;
    val = enumCheck((int)Class2.Struct1.Enum13.Val1, 4131);
    val = enumCheck((int)Class2.Struct1.Enum13.Val2, 4132);
    val = enumCheck((int)Class2.Struct1.Enum13.Val3, val);
    val = enumCheck((int)Class2.Struct1.Enum13.Val4, val);
    val = enumCheck((int)Class2.Struct1.Enum13.Val5l, 4131);
    val = enumCheck((int)Class2.Struct1.Enum13.Val6l, val);

    val = 0;
    val = enumCheck((int)Class2.Struct1.Enum14.Val1, 4141);
    val = enumCheck((int)Class2.Struct1.Enum14.Val2, 4142);
    val = enumCheck((int)Class2.Struct1.Enum14.Val3, val);
    val = enumCheck((int)Class2.Struct1.Enum14.Val4, val);
    val = enumCheck((int)Class2.Struct1.Enum14.Val5m, 4141);
    val = enumCheck((int)Class2.Struct1.Enum14.Val6m, val);

    Class1 class1 = new Class1();
    enumCheck((int)class1.class1Test1(Enum1.Val5a), 13);
    enumCheck((int)class1.class1Test2(Class1.Enum12.Val5c), 1121);
    enumCheck((int)class1.class1Test3(Class1.Struct1.Enum12.Val5f), 3121);

    enumCheck((int)cpp11_strongly_typed_enumerations.globalTest1(Enum1.Val5a), 13);
    enumCheck((int)cpp11_strongly_typed_enumerations.globalTest2(Class1.Enum12.Val5c), 1121);
    enumCheck((int)cpp11_strongly_typed_enumerations.globalTest3(Class1.Struct1.Enum12.Val5f), 3121);
  }
}

