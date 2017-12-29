
// This is the union runtime testcase. It ensures that values within a 
// union embedded within a struct can be set and read correctly.

import unions.*;

public class unions_runme {

  static {
    try {
	System.loadLibrary("unions");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) {

    // Create new instances of SmallStruct and BigStruct for later use
    SmallStruct small = new SmallStruct();
    small.setJill((short)200);

    BigStruct big = new BigStruct();
    big.setSmallstruct(small);
    big.setJack(300);

    // Use SmallStruct then BigStruct to setup EmbeddedUnionTest.
    // Ensure values in EmbeddedUnionTest are set correctly for each.
    EmbeddedUnionTest eut = new EmbeddedUnionTest();

    // First check the SmallStruct in EmbeddedUnionTest
    eut.setNumber(1);
    eut.getUni().setSmall(small);
    short Jill1 = eut.getUni().getSmall().getJill();
    if (Jill1 != 200) {
        throw new RuntimeException("Runtime test1 failed. eut.uni.small.jill=" + Jill1);
    }

    int Num1 = eut.getNumber();
    if (Num1 != 1) {
        throw new RuntimeException("Runtime test2 failed. eut.number=" + Num1);
    }

    // Secondly check the BigStruct in EmbeddedUnionTest
    eut.setNumber(2);
    eut.getUni().setBig(big);
    int Jack1 = eut.getUni().getBig().getJack();
    if (Jack1 != 300) {
        throw new RuntimeException("Runtime test3 failed. eut.uni.big.jack=" + Jack1);
    }

    short Jill2 = eut.getUni().getBig().getSmallstruct().getJill();
    if (Jill2 != 200) {
        throw new RuntimeException("Runtime test4 failed. eut.uni.big.smallstruct.jill=" + Jill2);
    }

    int Num2 = eut.getNumber();
    if (Num2 != 2) {
        throw new RuntimeException("Runtime test5 failed. eut.number=" + Num2);
    }
}
}
