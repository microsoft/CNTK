
import inherit_target_language.*;


public class inherit_target_language_runme {

  static {
    try {
	System.loadLibrary("inherit_target_language");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) {
    new Derived1().targetLanguageBaseMethod();
    new Derived2().targetLanguageBaseMethod();

    new MultipleDerived1().targetLanguageBaseMethod();
    new MultipleDerived2().targetLanguageBaseMethod();
    new MultipleDerived3().f();
    new MultipleDerived4().g();

    BaseX baseX = new BaseX();
    baseX.basex();
    baseX.targetLanguageBase2Method();

    DerivedX derivedX = new DerivedX();
    derivedX.basex();
    derivedX.derivedx();
    derivedX.targetLanguageBase2Method();
  }
}

