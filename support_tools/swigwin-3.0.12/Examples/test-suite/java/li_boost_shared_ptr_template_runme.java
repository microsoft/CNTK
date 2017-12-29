

import li_boost_shared_ptr_template.*;

public class li_boost_shared_ptr_template_runme {

  static {
    try {
	System.loadLibrary("li_boost_shared_ptr_template");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) {
    {
      BaseINTEGER b = new BaseINTEGER();
      DerivedINTEGER d = new DerivedINTEGER();
      if (b.bar() != 1)
        throw new RuntimeException("test 1");
      if (d.bar() != 2)
        throw new RuntimeException("test 2");
      if (li_boost_shared_ptr_template.bar_getter(b) != 1)
        throw new RuntimeException("test 3");
      if (li_boost_shared_ptr_template.bar_getter(d) != 2)
        throw new RuntimeException("test 4");
    }

    {
      BaseDefaultInt b = new BaseDefaultInt();
      DerivedDefaultInt d = new DerivedDefaultInt();
      DerivedDefaultInt2 d2 = new DerivedDefaultInt2();
      if (b.bar2() != 3)
        throw new RuntimeException("test 5");
      if (d.bar2() != 4)
        throw new RuntimeException("test 6");
      if (d2.bar2() != 4)
        throw new RuntimeException("test 6");
      if (li_boost_shared_ptr_template.bar2_getter(b) != 3)
        throw new RuntimeException("test 7");
      if (li_boost_shared_ptr_template.bar2_getter(d) != 4)
        throw new RuntimeException("test 8");
      if (li_boost_shared_ptr_template.bar2_getter(d2) != 4)
        throw new RuntimeException("test 8");
    }
  }
}

