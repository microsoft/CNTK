
import rename1.*;

public class rename1_runme {

  static {
    try {
	System.loadLibrary("rename1");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  // The code in main is the same for rename1_runme, rename2_runme, rename3_runme and renam4_runme
  public static void main(String argv[]) {
    {
      XYZInt xyz = new XYZInt();
      NotXYZInt notxyz = new NotXYZInt();
      xyz.opIntPtrA();
      xyz.opIntPtrB();
      xyz.opAnother2();
      xyz.opT2();
      xyz.tMethod2(0);
      xyz.tMethodNotXYZ2(notxyz);
      xyz.opNotXYZ2();
      xyz.opXYZ2();
    }
    {
      XYZDouble xyz = new XYZDouble();
      NotXYZDouble notxyz = new NotXYZDouble();
      xyz.opIntPtrA();
      xyz.opIntPtrB();
      xyz.opAnother1();
      xyz.opT1();
      xyz.tMethod1(0);
      xyz.tMethodNotXYZ1(notxyz);
      xyz.opNotXYZ1();
      xyz.opXYZ1();
    }
    {
      XYZKlass xyz = new XYZKlass();
      NotXYZKlass notxyz = new NotXYZKlass();
      xyz.opIntPtrA();
      xyz.opIntPtrB();
      xyz.opAnother3();
      xyz.opT3();
      xyz.tMethod3(new Klass());
      xyz.tMethodNotXYZ3(notxyz);
      xyz.opNotXYZ3();
      xyz.opXYZ3();
    }
    {
      XYZEnu xyz = new XYZEnu();
      NotXYZEnu notxyz = new NotXYZEnu();
      xyz.opIntPtrA();
      xyz.opIntPtrB();
      xyz.opAnother4();
      xyz.opT4();
      xyz.tMethod4(Enu.En1);
      xyz.tMethodNotXYZ4(notxyz);
      xyz.opNotXYZ4();
      xyz.opXYZ4();
    }
    {
      ABC abc = new ABC();
      abc.methodABC(abc);
      Klass k = new Klass();
      abc.methodKlass(k);
      ABC a = abc.opABC();
      k = abc.opKlass();
    }
  }
}

