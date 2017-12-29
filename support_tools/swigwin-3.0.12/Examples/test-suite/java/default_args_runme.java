
import default_args.*;

public class default_args_runme {
  static {
    try {
        System.loadLibrary("default_args");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) 
  {
    if (default_args.anonymous() != 7771)
      throw new RuntimeException("anonymous (1) failed");
    if (default_args.anonymous(1234) != 1234)
      throw new RuntimeException("anonymous (2) failed");

    if (default_args.booltest() != true)
      throw new RuntimeException("booltest (1) failed");
    if (default_args.booltest(true) != true)
      throw new RuntimeException("booltest (2) failed");
    if (default_args.booltest(false) != false)
      throw new RuntimeException("booltest (3) failed");

    EnumClass ec = new EnumClass();
    if (ec.blah() != true)
      throw new RuntimeException("EnumClass failed");

    if (default_args.casts1() != null)
      throw new RuntimeException("casts1 failed");

    if (!default_args.casts2().equals("Hello"))
      throw new RuntimeException("casts2 failed");

    if (!default_args.casts1("Ciao").equals("Ciao"))
      throw new RuntimeException("casts1 not default failed");

    if (default_args.chartest1() != 'x')
      throw new RuntimeException("chartest1 failed");

    if (default_args.chartest2() != '\0')
      throw new RuntimeException("chartest2 failed");

    if (default_args.chartest1('y') != 'y')
      throw new RuntimeException("chartest1 not default failed");

    if (default_args.chartest1('y') != 'y')
      throw new RuntimeException("chartest1 not default failed");

    if (default_args.reftest1() != 42)
      throw new RuntimeException("reftest1 failed");

    if (default_args.reftest1(400) != 400)
      throw new RuntimeException("reftest1 not default failed");

    if (!default_args.reftest2().equals("hello"))
      throw new RuntimeException("reftest2 failed");

    // rename
    Foo foo = new Foo();
    foo.newname(); 
    foo.newname(10); 
    foo.renamed3arg(10, 10.0); 
    foo.renamed2arg(10); 
    foo.renamed1arg(); 
     
    // exception specifications
    try {
      default_args.exceptionspec();
      throw new RuntimeException("exceptionspec 1 failed");
    } catch (RuntimeException e) {
    }
    try {
      default_args.exceptionspec(-1);
      throw new RuntimeException("exceptionspec 2 failed");
    } catch (RuntimeException e) {
    }
    try {
      default_args.exceptionspec(100);
      throw new RuntimeException("exceptionspec 3 failed");
    } catch (RuntimeException e) {
    }
    Except ex = new Except(false);
    try {
      ex.exspec();
      throw new RuntimeException("exspec 1 failed");
    } catch (RuntimeException e) {
    }
    try {
      ex.exspec(-1);
      throw new RuntimeException("exspec 2 failed");
    } catch (RuntimeException e) {
    }
    try {
      ex.exspec(100);
      throw new RuntimeException("exspec 3 failed");
    } catch (RuntimeException e) {
    }
    try {
      ex = new Except(true);
      throw new RuntimeException("Except constructor 1 failed");
    } catch (RuntimeException e) {
    }
    try {
      ex = new Except(true, -2);
      throw new RuntimeException("Except constructor 2 failed");
    } catch (RuntimeException e) {
    }

    // Default parameters in static class methods
    if (Statics.staticmethod() != 10+20+30)
      throw new RuntimeException("staticmethod 1 failed");
    if (Statics.staticmethod(100) != 100+20+30)
      throw new RuntimeException("staticmethod 2 failed");
    if (Statics.staticmethod(100,200,300) != 100+200+300)
      throw new RuntimeException("staticmethod 3 failed");


    Tricky tricky = new Tricky();
    if (tricky.privatedefault() != 200)
      throw new RuntimeException("privatedefault failed");
    if (tricky.protectedint() != 2000)
      throw new RuntimeException("protectedint failed");
    if (tricky.protecteddouble() != 987.654)
      throw new RuntimeException("protecteddouble failed");
    if (tricky.functiondefault() != 500)
      throw new RuntimeException("functiondefault failed");
    if (tricky.contrived() != 'X')
      throw new RuntimeException("contrived failed");

    if (default_args.constructorcall().getVal() != -1)
      throw new RuntimeException("constructorcall test 1 failed");

    if (default_args.constructorcall(new Klass(2222)).getVal() != 2222)
      throw new RuntimeException("constructorcall test 2 failed");

    if (default_args.constructorcall(new Klass()).getVal() != -1)
      throw new RuntimeException("constructorcall test 3 failed");

    // const methods 
    ConstMethods cm = new ConstMethods();
    if (cm.coo() != 20)
      throw new RuntimeException("coo test 1 failed");
    if (cm.coo(1.0) != 20)
      throw new RuntimeException("coo test 2 failed");
  }
}
