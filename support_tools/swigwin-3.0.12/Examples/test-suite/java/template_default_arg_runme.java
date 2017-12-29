

import template_default_arg.*;

public class template_default_arg_runme {

  static {
    try {
	System.loadLibrary("template_default_arg");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) {
    {
      Hello_int helloInt = new Hello_int();
      helloInt.foo(Hello_int.Hi.hi);
    }
    {
      X_int x = new X_int();
      if (x.meth(20.0, 200) != 200)
        throw new RuntimeException("X_int test 1 failed");
      if (x.meth(20) != 20)
        throw new RuntimeException("X_int test 2 failed");
      if (x.meth() != 0)
        throw new RuntimeException("X_int test 3 failed");
    }

    {
      Y_unsigned y = new Y_unsigned();
      if (y.meth(20.0, 200) != 200)
        throw new RuntimeException("Y_unsigned test 1 failed");
      if (y.meth(20) != 20)
        throw new RuntimeException("Y_unsigned test 2 failed");
      if (y.meth() != 0)
        throw new RuntimeException("Y_unsigned test 3 failed");
    }

    {
      X_longlong x = new X_longlong();
      x = new X_longlong(20.0);
      x = new X_longlong(20.0, 200L);
    }
    {
      X_int x = new X_int();
      x = new X_int(20.0);
      x = new X_int(20.0, 200);
    }
    {
      X_hello_unsigned x = new X_hello_unsigned();
      x = new X_hello_unsigned(20.0);
      x = new X_hello_unsigned(20.0, new Hello_int());
    }
    {
      Y_hello_unsigned y = new Y_hello_unsigned();
      y.meth(20.0, new Hello_int());
      y.meth(new Hello_int());
      y.meth();
    }

    {
      Foo_Z_8 fz = new Foo_Z_8();
      X_Foo_Z_8 x = new X_Foo_Z_8();
      Foo_Z_8 fzc = x.meth(fz);
    }

    // Templated functions
    {
      // plain function: int ott(Foo<int>)
      if (template_default_arg.ott(new Foo_int()) != 30)
        throw new RuntimeException("ott test 1 failed");

      // %template(ott) ott<int, int>;
      if (template_default_arg.ott() != 10)
        throw new RuntimeException("ott test 2 failed");
      if (template_default_arg.ott(1) != 10)
        throw new RuntimeException("ott test 3 failed");
      if (template_default_arg.ott(1, 1) != 10)
        throw new RuntimeException("ott test 4 failed");

      if (template_default_arg.ott("hi") != 20)
        throw new RuntimeException("ott test 5 failed");
      if (template_default_arg.ott("hi", 1) != 20)
        throw new RuntimeException("ott test 6 failed");
      if (template_default_arg.ott("hi", 1, 1) != 20)
        throw new RuntimeException("ott test 7 failed");

      // %template(ott) ott<const char *>;
      if (template_default_arg.ottstring(new Hello_int(), "hi") != 40)
        throw new RuntimeException("ott test 8 failed");

      if (template_default_arg.ottstring(new Hello_int()) != 40)
        throw new RuntimeException("ott test 9 failed");

      // %template(ott) ott<int>;
      if (template_default_arg.ottint(new Hello_int(), 1) != 50)
        throw new RuntimeException("ott test 10 failed");

      if (template_default_arg.ottint(new Hello_int()) != 50)
        throw new RuntimeException("ott test 11 failed");

      // %template(ott) ott<double>;
      if (template_default_arg.ott(new Hello_int(), 1.0) != 60)
        throw new RuntimeException("ott test 12 failed");

      if (template_default_arg.ott(new Hello_int()) != 60)
        throw new RuntimeException("ott test 13 failed");
    }

    // Above test in namespaces
    {
      // plain function: int nsott(Foo<int>)
      if (template_default_arg.nsott(new Foo_int()) != 130)
        throw new RuntimeException("nsott test 1 failed");

      // %template(nsott) nsott<int, int>;
      if (template_default_arg.nsott() != 110)
        throw new RuntimeException("nsott test 2 failed");
      if (template_default_arg.nsott(1) != 110)
        throw new RuntimeException("nsott test 3 failed");
      if (template_default_arg.nsott(1, 1) != 110)
        throw new RuntimeException("nsott test 4 failed");

      if (template_default_arg.nsott("hi") != 120)
        throw new RuntimeException("nsott test 5 failed");
      if (template_default_arg.nsott("hi", 1) != 120)
        throw new RuntimeException("nsott test 6 failed");
      if (template_default_arg.nsott("hi", 1, 1) != 120)
        throw new RuntimeException("nsott test 7 failed");

      // %template(nsott) nsott<const char *>;
      if (template_default_arg.nsottstring(new Hello_int(), "hi") != 140)
        throw new RuntimeException("nsott test 8 failed");

      if (template_default_arg.nsottstring(new Hello_int()) != 140)
        throw new RuntimeException("nsott test 9 failed");

      // %template(nsott) nsott<int>;
      if (template_default_arg.nsottint(new Hello_int(), 1) != 150)
        throw new RuntimeException("nsott test 10 failed");

      if (template_default_arg.nsottint(new Hello_int()) != 150)
        throw new RuntimeException("nsott test 11 failed");

      // %template(nsott) nsott<double>;
      if (template_default_arg.nsott(new Hello_int(), 1.0) != 160)
        throw new RuntimeException("nsott test 12 failed");

      if (template_default_arg.nsott(new Hello_int()) != 160)
        throw new RuntimeException("nsott test 13 failed");
    }
  }
}

