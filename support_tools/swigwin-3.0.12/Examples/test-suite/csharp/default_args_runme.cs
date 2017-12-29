using System;
using default_argsNamespace;

public class runme
{
    static void Main() 
    {
      if (default_args.anonymous() != 7771)
        throw new Exception("anonymous (1) failed");
      if (default_args.anonymous(1234) != 1234)
        throw new Exception("anonymous (2) failed");

      if (default_args.booltest() != true)
        throw new Exception("booltest (1) failed");
      if (default_args.booltest(true) != true)
        throw new Exception("booltest (2) failed");
      if (default_args.booltest(false) != false)
        throw new Exception("booltest (3) failed");

      EnumClass ec = new EnumClass();
      if (ec.blah() != true)
        throw new Exception("EnumClass failed");

      if (default_args.casts1() != null)
        throw new Exception("casts1 failed");

      if (default_args.casts2() != "Hello")
        throw new Exception("casts2 failed");

      if (default_args.casts1("Ciao") != "Ciao")
        throw new Exception("casts1 not default failed");

      if (default_args.chartest1() != 'x')
        throw new Exception("chartest1 failed");

      if (default_args.chartest2() != '\0')
        throw new Exception("chartest2 failed");

      if (default_args.chartest1('y') != 'y')
        throw new Exception("chartest1 not default failed");

      if (default_args.chartest1('y') != 'y')
        throw new Exception("chartest1 not default failed");

      if (default_args.reftest1() != 42)
        throw new Exception("reftest1 failed");

      if (default_args.reftest1(400) != 400)
        throw new Exception("reftest1 not default failed");

      if (default_args.reftest2() != "hello")
        throw new Exception("reftest2 failed");

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
        throw new Exception("exceptionspec 1 failed");
      } catch (Exception) {
      }
      try {
        default_args.exceptionspec(-1);
        throw new Exception("exceptionspec 2 failed");
      } catch (Exception) {
      }
      try {
        default_args.exceptionspec(100);
        throw new Exception("exceptionspec 3 failed");
      } catch (Exception) {
      }
      Except ex = new Except(false);
      try {
        ex.exspec();
        throw new Exception("exspec 1 failed");
      } catch (Exception) {
      }
      try {
        ex.exspec(-1);
        throw new Exception("exspec 2 failed");
      } catch (Exception) {
      }
      try {
        ex.exspec(100);
        throw new Exception("exspec 3 failed");
      } catch (Exception) {
      }
      try {
        ex = new Except(true);
        throw new Exception("Except constructor 1 failed");
      } catch (Exception) {
      }
      try {
        ex = new Except(true, -2);
        throw new Exception("Except constructor 2 failed");
      } catch (Exception) {
      }

      // Default parameters in static class methods
      if (Statics.staticmethod() != 10+20+30)
        throw new Exception("staticmethod 1 failed");
      if (Statics.staticmethod(100) != 100+20+30)
        throw new Exception("staticmethod 2 failed");
      if (Statics.staticmethod(100,200,300) != 100+200+300)
        throw new Exception("staticmethod 3 failed");


      Tricky tricky = new Tricky();
      if (tricky.privatedefault() != 200)
        throw new Exception("privatedefault failed");
      if (tricky.protectedint() != 2000)
        throw new Exception("protectedint failed");
      if (tricky.protecteddouble() != 987.654)
        throw new Exception("protecteddouble failed");
      if (tricky.functiondefault() != 500)
        throw new Exception("functiondefault failed");
      if (tricky.contrived() != 'X')
        throw new Exception("contrived failed");

      if (default_args.constructorcall().val != -1)
        throw new Exception("constructorcall test 1 failed");

      if (default_args.constructorcall(new Klass(2222)).val != 2222)
        throw new Exception("constructorcall test 2 failed");

      if (default_args.constructorcall(new Klass()).val != -1)
        throw new Exception("constructorcall test 3 failed");

      // const methods 
      ConstMethods cm = new ConstMethods();
      if (cm.coo() != 20)
        throw new Exception("coo test 1 failed");
      if (cm.coo(1.0) != 20)
        throw new Exception("coo test 2 failed");
    }
}

