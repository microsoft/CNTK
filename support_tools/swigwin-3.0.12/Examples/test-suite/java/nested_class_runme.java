
import nested_class.*;

public class nested_class_runme {

  static {
    try {
	System.loadLibrary("nested_class");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) {
    Outer outer = new Outer();
    Outer.InnerStruct1 is1 = outer.makeInnerStruct1();
    Outer.InnerClass1 ic1 = outer.makeInnerClass1();
    Outer.InnerUnion1 iu1 = outer.makeInnerUnion1();

    Outer.InnerStruct2 is2 = outer.makeInnerStruct2();
    Outer.InnerClass2 ic2 = outer.makeInnerClass2();
    Outer.InnerUnion2 iu2 = outer.makeInnerUnion2();

    Outer.InnerClass4Typedef ic4 = outer.makeInnerClass4Typedef();
    Outer.InnerStruct4Typedef is4 = outer.makeInnerStruct4Typedef();
    Outer.InnerUnion4Typedef iu4 = outer.makeInnerUnion4Typedef();

    Outer.InnerClass5Typedef ic5 = outer.makeInnerClass5();
    Outer.InnerStruct5Typedef is5 = outer.makeInnerStruct5();
    Outer.InnerUnion5Typedef iu5 = outer.makeInnerUnion5();

    ic5 = outer.makeInnerClass5Typedef();
    is5 = outer.makeInnerStruct5Typedef();
    iu5 = outer.makeInnerUnion5Typedef();

    {
      Outer.InnerMultiple im1 = outer.getMultipleInstance1();
      Outer.InnerMultiple im2 = outer.getMultipleInstance2();
      Outer.InnerMultiple im3 = outer.getMultipleInstance3();
      Outer.InnerMultiple im4 = outer.getMultipleInstance4();
    }

    {
      Outer.InnerMultipleDerived im1 = outer.getMultipleDerivedInstance1();
      Outer.InnerMultipleDerived im2 = outer.getMultipleDerivedInstance2();
      Outer.InnerMultipleDerived im3 = outer.getMultipleDerivedInstance3();
      Outer.InnerMultipleDerived im4 = outer.getMultipleDerivedInstance4();
    }

    {
      Outer.InnerMultipleDerived im1 = outer.getMultipleDerivedInstance1();
      Outer.InnerMultipleDerived im2 = outer.getMultipleDerivedInstance2();
      Outer.InnerMultipleDerived im3 = outer.getMultipleDerivedInstance3();
      Outer.InnerMultipleDerived im4 = outer.getMultipleDerivedInstance4();
    }

    {
      Outer.InnerMultipleAnonTypedef1 mat1 = outer.makeInnerMultipleAnonTypedef1();
      Outer.InnerMultipleAnonTypedef1 mat2 = outer.makeInnerMultipleAnonTypedef2();
      SWIGTYPE_p_p_Outer__InnerMultipleAnonTypedef1 mat3 = outer.makeInnerMultipleAnonTypedef3();

      Outer.InnerMultipleNamedTypedef1 mnt = outer.makeInnerMultipleNamedTypedef();
      Outer.InnerMultipleNamedTypedef1 mnt1 = outer.makeInnerMultipleNamedTypedef1();
      Outer.InnerMultipleNamedTypedef1 mnt2 = outer.makeInnerMultipleNamedTypedef2();
      SWIGTYPE_p_p_Outer__InnerMultipleNamedTypedef mnt3 = outer.makeInnerMultipleNamedTypedef3();
    }
    {
      Outer.InnerSameName isn = outer.makeInnerSameName();
    }
  }
}
