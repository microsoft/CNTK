using System;
using inherit_target_languageNamespace;

public class inherit_target_language_runme {
  public static void Main() {
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

