
import rname.*;

public class rname_runme {

  static {
    try {
	System.loadLibrary("rname");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) {
    rname.foo_i(10);
    rname.foo_d(10.0);
    rname.foo_s((short)10);
    rname.foo((long)10);

    Bar bar = new Bar();
    bar.foo_i(10);
    bar.foo_d(10.0);
    bar.foo((short)10);
    bar.foo_u((long)10);

    RenamedBase base = new RenamedBase();
    base.fn(base, base, base);
    if (!base.newname(10.0).equals("Base"))
      throw new RuntimeException("base.newname");

    RenamedDerived derived = new RenamedDerived();
    derived.Xfunc(base, base, base);
    if (!derived.newname(10.0).equals("Derived"))
      throw new RuntimeException("derived.newname");
  }
}

