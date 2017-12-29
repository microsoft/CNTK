import template_partial_specialization.*;

public class template_partial_specialization_runme {

  static {
    try {
	System.loadLibrary("template_partial_specialization");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) {
    // One parameter tests
    new A().a();
    new B().b();
    new C().c();
    new D().d();
    new E().e();

    new F().f();
    new G().g();
    new H().h();

    new I().i();
    new J().j();
    new K().k();
    new L().l();

    new BB().b();
    new BBB().b();
    new BBBB().b();
    new BBBBB().b();

    new B1().b();
    new B2().b();
    new B3().b();
    new B4().b();

    // Two parameter tests
    new A_().a();
    new B_().b();
    new C_().c();
    new D_().d();
    new E_().e();
    new F_().f();
    new G_().g();

    new C1_().c();
    new C2_().c();
    new C3_().c();
    new C4_().c();
    new B1_().b();
    new E1_().e();
    new E2_().e();
  }
}

