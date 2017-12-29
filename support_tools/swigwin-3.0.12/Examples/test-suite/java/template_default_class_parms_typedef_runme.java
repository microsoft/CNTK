

import template_default_class_parms_typedef.*;

public class template_default_class_parms_typedef_runme {

  static {
    try {
	System.loadLibrary("template_default_class_parms_typedef");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) {
    {
      DefaultBar bar = new DefaultBar(20.0, new SomeType(), 10);
      double d = bar.getCType();
      bar.setCType(d);
      SomeType s = bar.getDType();
      bar.setDType(s);
      int i = bar.getEType();
      bar.setEType(i);
      d = bar.method(d, s, i);
      d = bar.method_1(d, s, i);
      d = bar.method_2(d, s, i);
      d = bar.method_3(d, s, i);

      bar = new DefaultBar(true, 20.0, new SomeType(), 10);
      bar = new DefaultBar(true, true, 20.0, new SomeType(), 10);
      bar = new DefaultBar(true, true, true, 20.0, new SomeType(), 10);
    }
    {
      DefaultFoo foo = new DefaultFoo(new SomeType());
      SomeType s = foo.getTType();
      s = foo.method(s);
      s = foo.method_A(s);
      s = foo.method_B(s);
      s = foo.method_C(s);

      foo = new DefaultFoo(new SomeType(), new SomeType());
      foo = new DefaultFoo(new SomeType(), new SomeType(), new SomeType());
      foo = new DefaultFoo(new SomeType(), new SomeType(), new SomeType(), new SomeType());
    }
    {
      BarAnotherTypeBool bar = new BarAnotherTypeBool(new AnotherType(), true, 10);
      AnotherType a = bar.getCType();
      bar.setCType(a);
      boolean b = bar.getDType();
      bar.setDType(b);
      int i = bar.getEType();
      bar.setEType(i);

      a = bar.method(a, b, i);
      a = bar.method_1(a, b, i);
      a = bar.method_2(a, b, i);
      a = bar.method_3(a, b, i);

      bar = new BarAnotherTypeBool(true, new AnotherType(), true, 10);
      bar = new BarAnotherTypeBool(true, true, new AnotherType(), true, 10);
      bar = new BarAnotherTypeBool(true, true, true, new AnotherType(), true, 10);
    }
    {
      FooAnotherType foo = new FooAnotherType(new AnotherType());
      AnotherType a = foo.getTType();
      foo.setTType(a);
      a = foo.method(a);
      a = foo.method_A(a);
      a = foo.method_B(a);
      a = foo.method_C(a);

      foo = new FooAnotherType(new AnotherType(), new AnotherType());
      foo = new FooAnotherType(new AnotherType(), new AnotherType(), new AnotherType());
      foo = new FooAnotherType(new AnotherType(), new AnotherType(), new AnotherType(), new AnotherType());
    }
    {
      UsesBarDouble u = new UsesBarDouble();
      u.use_A(10.1, new SomeType(), 10);
      u.use_B(10.1, new SomeType(), 10);
      u.use_C(10.1, new SomeType(), 10);
      u.use_D(10.1, new SomeType(), 10);
    }
  }
}

