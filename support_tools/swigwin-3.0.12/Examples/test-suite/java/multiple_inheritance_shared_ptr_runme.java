import multiple_inheritance_shared_ptr.*;

public class multiple_inheritance_shared_ptr_runme {

  static {
    try {
      System.loadLibrary("multiple_inheritance_shared_ptr");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  //Test base class as a parameter in java
  int jcbase1b(CBase1 cb1){
    return cb1.cbase1y();
  }
  int jabase1(ABase1 ab1){
    return ab1.abase1();
  }
  int jcbase2(CBase2 cb2){
    return cb2.cbase2();
  }

  public static void check(boolean fail, String msg) {
    if (fail)
      throw new RuntimeException(msg);
  }

  public static void main(String argv[]) {
    //Test Derived1
    Derived1 d1=new Derived1();
    check(d1.cbase1y()!=3, "Derived1::cbase1y() failed");
    check(d1.cbase2()!=4, "Derived1::cbase2() failed");

    //Test Derived2
    Derived2 d2=new Derived2();
    check(d2.cbase1y()!=6, "Derived2::cbase1y() failed");
    check(d2.abase1()!=5, "Derived2::abase1() failed");

    //Test Derived3
    Derived3 d3=new Derived3();
    check(d3.cbase1y()!=7, "Derived3::cbase1y() failed");
    check(d3.cbase2()!=8, "Derived3::cbase2() failed");
    check(d3.abase1()!=9, "Derived3::abase1() failed");

    //Test Bottom1
    Bottom1 b1=new Bottom1();
    check(b1.cbase1y()!=103, "Bottom1::cbase1y() failed");
    check(b1.cbase2()!=104, "Bottom1::cbase2() failed");

    //Test Bottom2
    Bottom2 b2=new Bottom2();
    check(b2.cbase1y()!=206, "Bottom2::cbase1y() failed");
    check(b2.abase1()!=205, "Bottom2::abase1() failed");

    //Test Bottom3
    Bottom3 b3=new Bottom3();
    check(b3.cbase1y()!=307, "Bottom3::cbase1y() failed");
    check(b3.cbase2()!=308, "Bottom3::cbase2() failed");
    check(b3.abase1()!=309, "Bottom3::abase1() failed");

    //Test interfaces from c++ classes 
    CBase1 cb1=new CBase1SwigImpl();
    CBase2 cb2=new CBase2SwigImpl();
    check(cb1.cbase1y()!=1, "CBase1::cbase1y() failed");
    check(cb2.cbase2()!=2, "CBase2::cbase2() failed");

    //Test abstract class as return value
    ABase1 ab1=d3.cloneit();
    check(ab1.abase1()!=9, "Derived3::abase1() through ABase1 failed");

    //Test concrete base class as return value
    CBase1 cb6=d2.cloneit();
    CBase2 cb7=d1.cloneit();
    check(cb6.cbase1y()!=6, "Derived2::cbase1y() through CBase1 failed");
    check(cb7.cbase2()!=4, "Derived1:cbase2() through ABase1 failed");

    //Test multi inheritance 
    CBase1 cb3=new Derived1();
    CBase1 cb4=new Derived3();
    CBase2 cb5=new Derived3();
    ABase1 ab6=new Derived2();
    check(cb3.cbase1y()!=3, "Derived1::cbase1y() through CBase1 failed");
    check(cb4.cbase1y()!=7, "Derived3::cbase1y() through CBase1 failed");
    check(cb5.cbase2()!=8, "Derived3::cbase2() through CBase2 failed");
    check(ab6.abase1()!=5, "Derived2::abase1() through ABase1 failed");  

    //Test base classes as parameter in java 
    multiple_inheritance_shared_ptr_runme mhar=new multiple_inheritance_shared_ptr_runme();
    check(mhar.jcbase1b(d1)!=3, "jcbase1b() through Derived1 as parameter failed");
    check(mhar.jcbase1b(d2)!=6, "jcbase1b() through Derived2 as parameter failed");
    check(mhar.jcbase1b(d3)!=7, "jcbase1b() through Derived3 as parameter failed");
    check(mhar.jcbase2(d1)!=4, "jcbase2() through Derived1 as parameter failed");
    check(mhar.jcbase2(d3)!=8, "jcbase2() through Derived3 as parameter failed");
    check(mhar.jabase1(d2)!=5, "jabase1() through Derived2 as parameter failed");
    check(mhar.jabase1(d3)!=9, "jabase1() through Derived3 as parameter failed");

    //Value parameters
    //Test CBase1 CBase2 as parameters (note slicing for Derived and Bottom classes)
    check(multiple_inheritance_shared_ptr.InputValCBase1(d1)!=1, "InputValCBase1(), Derived1 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputValCBase1(d2)!=1, "InputValCBase1(), Derived2 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputValCBase1(d3)!=1, "InputValCBase1(), Derived3 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputValCBase2(d3)!=2, "InputValCBase2(), Derived3 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputValCBase2(d1)!=2, "InputValCBase2(), Derived1 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputValCBase1(cb1)!=1, "InputValCBase1(), CBase1 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputValCBase2(cb2)!=2, "InputValCBase2(), CBase2 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputValCBase1(b1)!=1, "InputValCBase1(), Bottom1 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputValCBase1(b2)!=1, "InputValCBase1(), Bottom2 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputValCBase1(b3)!=1, "InputValCBase1(), Bottom3 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputValCBase2(b3)!=2, "InputValCBase2(), Bottom3 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputValCBase2(b1)!=2, "InputValCBase2(), Bottom1 as a parameter failed");

    //Pointer parameters
    //Test ABase1 as a parameter
    check(multiple_inheritance_shared_ptr.InputPtrABase1(d2)!=5, "InputPtrABase1() through Derived2 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputPtrABase1(d3)!=9, "InputPtrABase1() through Derived3 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputPtrABase1(b2)!=205, "InputPtrABase1() through Bottom2 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputPtrABase1(b3)!=309, "InputPtrABase1() through Bottom3 as a parameter failed");

    //Test CBase1 CBase2 as parameters
    check(multiple_inheritance_shared_ptr.InputPtrCBase1(d1)!=3, "InputPtrCBase1(), Derived1 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputPtrCBase1(d2)!=6, "InputPtrCBase1(), Derived2 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputPtrCBase1(d3)!=7, "InputPtrCBase1(), Derived3 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputPtrCBase2(d3)!=8, "InputPtrCBase2(), Derived3 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputPtrCBase2(d1)!=4, "InputPtrCBase2(), Derived1 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputPtrCBase1(cb1)!=1, "InputPtrCBase1(), CBase1 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputPtrCBase2(cb2)!=2, "InputPtrCBase2(), CBase2 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputPtrCBase1(b1)!=103, "InputPtrCBase1(), Bottom1 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputPtrCBase1(b2)!=206, "InputPtrCBase1(), Bottom2 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputPtrCBase1(b3)!=307, "InputPtrCBase1(), Bottom3 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputPtrCBase2(b3)!=308, "InputPtrCBase2(), Bottom3 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputPtrCBase2(b1)!=104, "InputPtrCBase2(), Bottom1 as a parameter failed");

    //Reference parameters
    //Test ABase1 as a parameter
    check(multiple_inheritance_shared_ptr.InputRefABase1(d2)!=5, "InputRefABase1() through Derived2 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputRefABase1(d3)!=9, "InputRefABase1() through Derived3 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputRefABase1(b2)!=205, "InputRefABase1() through Bottom2 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputRefABase1(b3)!=309, "InputRefABase1() through Bottom3 as a parameter failed");

    //Test CBase1 CBase2 as parameters
    check(multiple_inheritance_shared_ptr.InputRefCBase1(d1)!=3, "InputRefCBase1(), Derived1 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputRefCBase1(d2)!=6, "InputRefCBase1(), Derived2 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputRefCBase1(d3)!=7, "InputRefCBase1(), Derived3 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputRefCBase2(d3)!=8, "InputRefCBase2(), Derived3 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputRefCBase2(d1)!=4, "InputRefCBase2(), Derived1 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputRefCBase1(cb1)!=1, "InputRefCBase1(), CBase1 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputRefCBase2(cb2)!=2, "InputRefCBase2(), CBase2 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputRefCBase1(b1)!=103, "InputRefCBase1(), Bottom1 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputRefCBase1(b2)!=206, "InputRefCBase1(), Bottom2 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputRefCBase1(b3)!=307, "InputRefCBase1(), Bottom3 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputRefCBase2(b3)!=308, "InputRefCBase2(), Bottom3 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputRefCBase2(b1)!=104, "InputRefCBase2(), Bottom1 as a parameter failed");

    //Const reference pointer parameters
    //Test ABase1 as a parameter
    check(multiple_inheritance_shared_ptr.InputCPtrRefABase1(d2)!=5, "InputCPtrRefABase1() through Derived2 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputCPtrRefABase1(d3)!=9, "InputCPtrRefABase1() through Derived3 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputCPtrRefABase1(b2)!=205, "InputCPtrRefABase1() through Bottom2 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputCPtrRefABase1(b3)!=309, "InputCPtrRefABase1() through Bottom3 as a parameter failed");

    //Test CBase1 CBase2 as parameters
    check(multiple_inheritance_shared_ptr.InputCPtrRefCBase1(d1)!=3, "InputCPtrRefCBase1(), Derived1 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputCPtrRefCBase1(d2)!=6, "InputCPtrRefCBase1(), Derived2 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputCPtrRefCBase1(d3)!=7, "InputCPtrRefCBase1(), Derived3 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputCPtrRefCBase2(d3)!=8, "InputCPtrRefCBase2(), Derived3 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputCPtrRefCBase2(d1)!=4, "InputCPtrRefCBase2(), Derived1 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputCPtrRefCBase1(cb1)!=1, "InputCPtrRefCBase1(), CBase1 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputCPtrRefCBase2(cb2)!=2, "InputCPtrRefCBase2(), CBase2 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputCPtrRefCBase1(b1)!=103, "InputCPtrRefCBase1(), Bottom1 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputCPtrRefCBase1(b2)!=206, "InputCPtrRefCBase1(), Bottom2 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputCPtrRefCBase1(b3)!=307, "InputCPtrRefCBase1(), Bottom3 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputCPtrRefCBase2(b3)!=308, "InputCPtrRefCBase2(), Bottom3 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputCPtrRefCBase2(b1)!=104, "InputCPtrRefCBase2(), Bottom1 as a parameter failed");

    //Shared pointer parameters
    //Test ABase1 as a parameter
    check(multiple_inheritance_shared_ptr.InputSharedPtrABase1(d2)!=5, "InputSharedPtrABase1() through Derived2 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputSharedPtrABase1(d3)!=9, "InputSharedPtrABase1() through Derived3 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputSharedPtrABase1(b2)!=205, "InputSharedPtrABase1() through Bottom2 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputSharedPtrABase1(b3)!=309, "InputSharedPtrABase1() through Bottom3 as a parameter failed");

    //Test CBase1 CBase2 as parameters
    check(multiple_inheritance_shared_ptr.InputSharedPtrCBase1(d1)!=3, "InputSharedPtrCBase1(), Derived1 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputSharedPtrCBase1(d2)!=6, "InputSharedPtrCBase1(), Derived2 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputSharedPtrCBase1(d3)!=7, "InputSharedPtrCBase1(), Derived3 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputSharedPtrCBase2(d3)!=8, "InputSharedPtrCBase2(), Derived3 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputSharedPtrCBase2(d1)!=4, "InputSharedPtrCBase2(), Derived1 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputSharedPtrCBase1(cb1)!=1, "InputSharedPtrCBase1(), CBase1 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputSharedPtrCBase2(cb2)!=2, "InputSharedPtrCBase2(), CBase2 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputSharedPtrCBase1(b1)!=103, "InputSharedPtrCBase1(), Bottom1 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputSharedPtrCBase1(b2)!=206, "InputSharedPtrCBase1(), Bottom2 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputSharedPtrCBase1(b3)!=307, "InputSharedPtrCBase1(), Bottom3 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputSharedPtrCBase2(b3)!=308, "InputSharedPtrCBase2(), Bottom3 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputSharedPtrCBase2(b1)!=104, "InputSharedPtrCBase2(), Bottom1 as a parameter failed");

    //Shared pointer reference parameters
    //Test ABase1 as a parameter
    check(multiple_inheritance_shared_ptr.InputSharedPtrRefABase1(d2)!=5, "InputSharedPtrRefABase1() through Derived2 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputSharedPtrRefABase1(d3)!=9, "InputSharedPtrRefABase1() through Derived3 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputSharedPtrRefABase1(b2)!=205, "InputSharedPtrRefABase1() through Bottom2 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputSharedPtrRefABase1(b3)!=309, "InputSharedPtrRefABase1() through Bottom3 as a parameter failed");

    //Test CBase1 CBase2 as parameters
    check(multiple_inheritance_shared_ptr.InputSharedPtrRefCBase1(d1)!=3, "InputSharedPtrRefCBase1(), Derived1 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputSharedPtrRefCBase1(d2)!=6, "InputSharedPtrRefCBase1(), Derived2 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputSharedPtrRefCBase1(d3)!=7, "InputSharedPtrRefCBase1(), Derived3 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputSharedPtrRefCBase2(d3)!=8, "InputSharedPtrRefCBase2(), Derived3 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputSharedPtrRefCBase2(d1)!=4, "InputSharedPtrRefCBase2(), Derived1 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputSharedPtrRefCBase1(cb1)!=1, "InputSharedPtrRefCBase1(), CBase1 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputSharedPtrRefCBase2(cb2)!=2, "InputSharedPtrRefCBase2(), CBase2 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputSharedPtrRefCBase1(b1)!=103, "InputSharedPtrRefCBase1(), Bottom1 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputSharedPtrRefCBase1(b2)!=206, "InputSharedPtrRefCBase1(), Bottom2 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputSharedPtrRefCBase1(b3)!=307, "InputSharedPtrRefCBase1(), Bottom3 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputSharedPtrRefCBase2(b3)!=308, "InputSharedPtrRefCBase2(), Bottom3 as a parameter failed");
    check(multiple_inheritance_shared_ptr.InputSharedPtrRefCBase2(b1)!=104, "InputSharedPtrRefCBase2(), Bottom1 as a parameter failed");

    //Derived classes as parameters
    check(multiple_inheritance_shared_ptr.InputValDerived1(d1)!=3+4, "InputValDerived1() failed");
    check(multiple_inheritance_shared_ptr.InputValDerived2(d2)!=6+5, "InputValDerived2() failed");
    check(multiple_inheritance_shared_ptr.InputValDerived3(d3)!=7+8+9, "InputValDerived3() failed");

    check(multiple_inheritance_shared_ptr.InputRefDerived1(d1)!=3+4, "InputRefDerived1() failed");
    check(multiple_inheritance_shared_ptr.InputRefDerived2(d2)!=6+5, "InputRefDerived2() failed");
    check(multiple_inheritance_shared_ptr.InputRefDerived3(d3)!=7+8+9, "InputRefDerived3() failed");

    check(multiple_inheritance_shared_ptr.InputPtrDerived1(d1)!=3+4, "InputPtrDerived1() failed");
    check(multiple_inheritance_shared_ptr.InputPtrDerived2(d2)!=6+5, "InputPtrDerived2() failed");
    check(multiple_inheritance_shared_ptr.InputPtrDerived3(d3)!=7+8+9, "InputPtrDerived3() failed");

    check(multiple_inheritance_shared_ptr.InputCPtrRefDerived1(d1)!=3+4, "InputCPtrRefDerived1() failed");
    check(multiple_inheritance_shared_ptr.InputCPtrRefDerived2(d2)!=6+5, "InputCPtrRefDerived2() failed");
    check(multiple_inheritance_shared_ptr.InputCPtrRefDerived3(d3)!=7+8+9, "InputCPtrRefDerived3() failed");

    check(multiple_inheritance_shared_ptr.InputSharedPtrDerived1(d1)!=3+4, "InputSharedPtrDerived1() failed");
    check(multiple_inheritance_shared_ptr.InputSharedPtrDerived2(d2)!=6+5, "InputSharedPtrDerived2() failed");
    check(multiple_inheritance_shared_ptr.InputSharedPtrDerived3(d3)!=7+8+9, "InputSharedPtrDerived3() failed");

    check(multiple_inheritance_shared_ptr.InputSharedPtrRefDerived1(d1)!=3+4, "InputSharedPtrRefDerived1() failed");
    check(multiple_inheritance_shared_ptr.InputSharedPtrRefDerived2(d2)!=6+5, "InputSharedPtrRefDerived2() failed");
    check(multiple_inheritance_shared_ptr.InputSharedPtrRefDerived3(d3)!=7+8+9, "InputSharedPtrRefDerived3() failed");

    //Bottom classes as Derived parameters
    check(multiple_inheritance_shared_ptr.InputValDerived1(b1)!=3+4, "InputValDerived1() failed");
    check(multiple_inheritance_shared_ptr.InputValDerived2(b2)!=6+5, "InputValDerived2() failed");
    check(multiple_inheritance_shared_ptr.InputValDerived3(b3)!=7+8+9, "InputValDerived3() failed");

    check(multiple_inheritance_shared_ptr.InputRefDerived1(b1)!=103+104, "InputRefDerived1() failed");
    check(multiple_inheritance_shared_ptr.InputRefDerived2(b2)!=206+205, "InputRefDerived2() failed");
    check(multiple_inheritance_shared_ptr.InputRefDerived3(b3)!=307+308+309, "InputRefDerived3() failed");

    check(multiple_inheritance_shared_ptr.InputPtrDerived1(b1)!=103+104, "InputPtrDerived1() failed");
    check(multiple_inheritance_shared_ptr.InputPtrDerived2(b2)!=206+205, "InputPtrDerived2() failed");
    check(multiple_inheritance_shared_ptr.InputPtrDerived3(b3)!=307+308+309, "InputPtrDerived3() failed");

    check(multiple_inheritance_shared_ptr.InputCPtrRefDerived1(b1)!=103+104, "InputCPtrRefDerived1() failed");
    check(multiple_inheritance_shared_ptr.InputCPtrRefDerived2(b2)!=206+205, "InputCPtrRefDerived2() failed");
    check(multiple_inheritance_shared_ptr.InputCPtrRefDerived3(b3)!=307+308+309, "InputCPtrRefDerived3() failed");

    check(multiple_inheritance_shared_ptr.InputSharedPtrDerived1(b1)!=103+104, "InputSharedPtrDerived1() failed");
    check(multiple_inheritance_shared_ptr.InputSharedPtrDerived2(b2)!=206+205, "InputSharedPtrDerived2() failed");
    check(multiple_inheritance_shared_ptr.InputSharedPtrDerived3(b3)!=307+308+309, "InputSharedPtrDerived3() failed");

    check(multiple_inheritance_shared_ptr.InputSharedPtrRefDerived1(b1)!=103+104, "InputSharedPtrRefDerived1() failed");
    check(multiple_inheritance_shared_ptr.InputSharedPtrRefDerived2(b2)!=206+205, "InputSharedPtrRefDerived2() failed");
    check(multiple_inheritance_shared_ptr.InputSharedPtrRefDerived3(b3)!=307+308+309, "InputSharedPtrRefDerived3() failed");

    //Bottom classes as Bottom parameters
    check(multiple_inheritance_shared_ptr.InputValBottom1(b1)!=103+104, "InputValBottom1() failed");
    check(multiple_inheritance_shared_ptr.InputValBottom2(b2)!=206+205, "InputValBottom2() failed");
    check(multiple_inheritance_shared_ptr.InputValBottom3(b3)!=307+308+309, "InputValBottom3() failed");

    check(multiple_inheritance_shared_ptr.InputRefBottom1(b1)!=103+104, "InputRefBottom1() failed");
    check(multiple_inheritance_shared_ptr.InputRefBottom2(b2)!=206+205, "InputRefBottom2() failed");
    check(multiple_inheritance_shared_ptr.InputRefBottom3(b3)!=307+308+309, "InputRefBottom3() failed");

    check(multiple_inheritance_shared_ptr.InputPtrBottom1(b1)!=103+104, "InputPtrBottom1() failed");
    check(multiple_inheritance_shared_ptr.InputPtrBottom2(b2)!=206+205, "InputPtrBottom2() failed");
    check(multiple_inheritance_shared_ptr.InputPtrBottom3(b3)!=307+308+309, "InputPtrBottom3() failed");

    check(multiple_inheritance_shared_ptr.InputCPtrRefBottom1(b1)!=103+104, "InputCPtrRefBottom1() failed");
    check(multiple_inheritance_shared_ptr.InputCPtrRefBottom2(b2)!=206+205, "InputCPtrRefBottom2() failed");
    check(multiple_inheritance_shared_ptr.InputCPtrRefBottom3(b3)!=307+308+309, "InputCPtrRefBottom3() failed");

    check(multiple_inheritance_shared_ptr.InputSharedPtrBottom1(b1)!=103+104, "InputSharedPtrBottom1() failed");
    check(multiple_inheritance_shared_ptr.InputSharedPtrBottom2(b2)!=206+205, "InputSharedPtrBottom2() failed");
    check(multiple_inheritance_shared_ptr.InputSharedPtrBottom3(b3)!=307+308+309, "InputSharedPtrBottom3() failed");

    check(multiple_inheritance_shared_ptr.InputSharedPtrRefBottom1(b1)!=103+104, "InputSharedPtrRefBottom1() failed");
    check(multiple_inheritance_shared_ptr.InputSharedPtrRefBottom2(b2)!=206+205, "InputSharedPtrRefBottom2() failed");
    check(multiple_inheritance_shared_ptr.InputSharedPtrRefBottom3(b3)!=307+308+309, "InputSharedPtrRefBottom3() failed");

    // Return pointers
    check(multiple_inheritance_shared_ptr.MakePtrDerived1_CBase1().cbase1y()!=3, "MakePtrDerived1_CBase1 failed");
    check(multiple_inheritance_shared_ptr.MakePtrDerived1_CBase2().cbase2()!=4, "MakePtrDerived1_CBase2 failed");
    check(multiple_inheritance_shared_ptr.MakePtrDerived2_CBase1().cbase1y()!=6, "MakePtrDerived2_CBase1 failed");
    check(multiple_inheritance_shared_ptr.MakePtrDerived2_ABase1().abase1()!=5, "MakePtrDerived2_ABase1 failed");
    check(multiple_inheritance_shared_ptr.MakePtrDerived3_ABase1().abase1()!=9, "MakePtrDerived3_ABase1 failed");
    check(multiple_inheritance_shared_ptr.MakePtrDerived3_CBase1().cbase1y()!=7, "MakePtrDerived3_CBase1 failed");
    check(multiple_inheritance_shared_ptr.MakePtrDerived3_CBase2().cbase2()!=8, "MakePtrDerived3_CBase2 failed");

    // Return references
    check(multiple_inheritance_shared_ptr.MakeRefDerived1_CBase1().cbase1y()!=3, "MakeRefDerived1_CBase1 failed");
    check(multiple_inheritance_shared_ptr.MakeRefDerived1_CBase2().cbase2()!=4, "MakeRefDerived1_CBase2 failed");
    check(multiple_inheritance_shared_ptr.MakeRefDerived2_CBase1().cbase1y()!=6, "MakeRefDerived2_CBase1 failed");
    check(multiple_inheritance_shared_ptr.MakeRefDerived2_ABase1().abase1()!=5, "MakeRefDerived2_ABase1 failed");
    check(multiple_inheritance_shared_ptr.MakeRefDerived3_ABase1().abase1()!=9, "MakeRefDerived3_ABase1 failed");
    check(multiple_inheritance_shared_ptr.MakeRefDerived3_CBase1().cbase1y()!=7, "MakeRefDerived3_CBase1 failed");
    check(multiple_inheritance_shared_ptr.MakeRefDerived3_CBase2().cbase2()!=8, "MakeRefDerived3_CBase2 failed");

    // Return by value (sliced objects)
    check(multiple_inheritance_shared_ptr.MakeValDerived1_CBase1().cbase1y()!=1, "MakeValDerived1_CBase1 failed");
    check(multiple_inheritance_shared_ptr.MakeValDerived1_CBase2().cbase2()!=2, "MakeValDerived1_CBase2 failed");
    check(multiple_inheritance_shared_ptr.MakeValDerived2_CBase1().cbase1y()!=1, "MakeValDerived2_CBase1 failed");
    check(multiple_inheritance_shared_ptr.MakeValDerived3_CBase1().cbase1y()!=1, "MakeValDerived3_CBase1 failed");
    check(multiple_inheritance_shared_ptr.MakeValDerived3_CBase2().cbase2()!=2, "MakeValDerived3_CBase2 failed");

    // Return smart pointers
    check(multiple_inheritance_shared_ptr.MakeSharedPtrDerived1_CBase1().cbase1y()!=3, "MakeSharedPtrDerived1_CBase1 failed");
    check(multiple_inheritance_shared_ptr.MakeSharedPtrDerived1_CBase2().cbase2()!=4, "MakeSharedPtrDerived1_CBase2 failed");
    check(multiple_inheritance_shared_ptr.MakeSharedPtrDerived2_CBase1().cbase1y()!=6, "MakeSharedPtrDerived2_CBase1 failed");
    check(multiple_inheritance_shared_ptr.MakeSharedPtrDerived2_ABase1().abase1()!=5, "MakeSharedPtrDerived2_ABase1 failed");
    check(multiple_inheritance_shared_ptr.MakeSharedPtrDerived3_ABase1().abase1()!=9, "MakeSharedPtrDerived3_ABase1 failed");
    check(multiple_inheritance_shared_ptr.MakeSharedPtrDerived3_CBase1().cbase1y()!=7, "MakeSharedPtrDerived3_CBase1 failed");
    check(multiple_inheritance_shared_ptr.MakeSharedPtrDerived3_CBase2().cbase2()!=8, "MakeSharedPtrDerived3_CBase2 failed");

    // Return smart pointers references
    check(multiple_inheritance_shared_ptr.MakeSharedPtrRefDerived1_CBase1().cbase1y()!=3, "MakeSharedPtrRefDerived1_CBase1 failed");
    check(multiple_inheritance_shared_ptr.MakeSharedPtrRefDerived1_CBase2().cbase2()!=4, "MakeSharedPtrRefDerived1_CBase2 failed");
    check(multiple_inheritance_shared_ptr.MakeSharedPtrRefDerived2_CBase1().cbase1y()!=6, "MakeSharedPtrRefDerived2_CBase1 failed");
    check(multiple_inheritance_shared_ptr.MakeSharedPtrRefDerived2_ABase1().abase1()!=5, "MakeSharedPtrRefDerived2_ABase1 failed");
    check(multiple_inheritance_shared_ptr.MakeSharedPtrRefDerived3_ABase1().abase1()!=9, "MakeSharedPtrRefDerived3_ABase1 failed");
    check(multiple_inheritance_shared_ptr.MakeSharedPtrRefDerived3_CBase1().cbase1y()!=7, "MakeSharedPtrRefDerived3_CBase1 failed");
    check(multiple_inheritance_shared_ptr.MakeSharedPtrRefDerived3_CBase2().cbase2()!=8, "MakeSharedPtrRefDerived3_CBase2 failed");
  }
}
