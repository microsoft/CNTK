import director_wombat.*;

public class director_wombat_runme
{
  static {
    try {
        System.loadLibrary("director_wombat");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String[] args)
  {
    Bar                 b = new Bar();
    Foo_integers        a;
    int                 retval;

    a = b.meth();
    if ((retval = a.meth(49)) != 49) {
      throw new RuntimeException ("Test failed: retval = " + retval + ", expected 49");
    }

    a.delete();

    a = new director_wombat_Foo_integers_derived();
    if ((retval = a.meth(62)) != 62 + 2) {
      throw new RuntimeException ("Test failed: retval = " + retval + ", expected 62 + 2");
    }

    a.delete();

    a = new director_wombat_Foo_integers_derived_2();
    if ((retval = a.meth(37)) != 37) {
      throw new RuntimeException ("Test failed: retval = " + retval + ", expected 37");
    }

    b.delete();

    b = new director_wombat_Bar_derived_1();
    b.foo_meth_ref(a, 0);
    b.foo_meth_ptr(a, 1);
    b.foo_meth_val(a, 2);
  }
}

class director_wombat_Foo_integers_derived extends Foo_integers
{
  public director_wombat_Foo_integers_derived()
  {
    super();
  }

  public int meth(int param)
  {
    return param + 2;
  }
}

class director_wombat_Foo_integers_derived_2 extends Foo_integers
{
  public director_wombat_Foo_integers_derived_2()
  {
    super();
  }
}

class director_wombat_Bar_derived_1 extends Bar
{
  public director_wombat_Bar_derived_1()
  {
    super();
  }

  public void foo_meth_ref(Foo_integers foo_obj, int param)
  {
    if (!(foo_obj instanceof director_wombat_Foo_integers_derived_2)) {
      throw new RuntimeException ("Test failed: foo_obj in foo_meth_ref is not director_wombat_Foo_integers_derived_2, got " + foo_obj);
    }
  }
  public void foo_meth_ptr(Foo_integers foo_obj, int param)
  {
    if (!(foo_obj instanceof director_wombat_Foo_integers_derived_2)) {
      throw new RuntimeException ("Test failed: foo_obj in foo_meth_ptr is not director_wombat_Foo_integers_derived_2, got " + foo_obj);
    }
  }
  public void foo_meth_val(Foo_integers foo_obj, int param)
  {
    if (!(foo_obj instanceof director_wombat_Foo_integers_derived_2)) {
      throw new RuntimeException ("Test failed: foo_obj in foo_meth_val is not director_wombat_Foo_integers_derived_2, got " + foo_obj);
    }
  }
}
