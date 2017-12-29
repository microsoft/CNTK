import operator_overload.*;

public class operator_overload_runme {

  static {
    System.loadLibrary("operator_overload");
  }

  public static void main(String argv[]) {

    // Java does not support operators, so we just check that these can be called as methods.

    Op.sanity_check();

    //test routine:
    Op a = new Op();
    Op b = new Op(5);
    Op c = new Op(b); // copy constructor
    Op d = new Op(2);
    Op dd = d.Equal(d); // assignment operator

    // test equality
    Assert(a.NotEqual(b));
    Assert(b.EqualEqual(c));
    Assert(a.NotEqual(d));
    Assert(d.EqualEqual(dd));

    // test <
    Assert(a.LessThan(b));
    Assert(a.LessThanEqual(b));
    Assert(b.LessThanEqual(c));
    Assert(b.GreaterThanEqual(c));
    Assert(b.GreaterThan(d));
    Assert(b.GreaterThanEqual(d));

    // test +=
    Op e = new Op(3);
    e.PlusEqual(d);
    Assert(e.EqualEqual(b));
    e.MinusEqual(c);
    Assert(e.EqualEqual(a));
    e = new Op(1);
    e.MultiplyEqual(b);
    Assert(e.EqualEqual(c));
    e.DivideEqual(d);
    Assert(e.EqualEqual(d));
    e.PercentEqual(c);
    Assert(e.EqualEqual(d));

    // test +
    Op f = new Op(1);
    Op g = new Op(1);
    Assert(f.Plus(g).EqualEqual(new Op(2)));
    Assert(f.Minus(g).EqualEqual(new Op(0)));
    Assert(f.Multiply(g).EqualEqual(new Op(1)));
    Assert(f.Divide(g).EqualEqual(new Op(1)));
    Assert(f.Percent(g).EqualEqual(new Op(0)));

    // test unary operators
    Assert((a.Not() == true));
    Assert((b.Not() == false));
    Assert(a.Minus().EqualEqual(a));
    Assert(b.Minus().EqualEqual( new Op(-5)));

    // test []
    Op h = new Op(3);
    Assert(h.__getitem__(0) == 3);
    Assert(h.__getitem__(1) == 0);
    h.__setitem__(0,2);	// set
    Assert(h.__getitem__(0) == 2);
    h.__setitem__(1,2);	// ignored
    Assert(h.IndexIntoConst(0) == 2);
    Assert(h.IndexIntoConst(1) == 0);

    // test ()
    Op i = new Op(3);
    Assert(i.Functor()==3);
    Assert(i.Functor(1)==4);
    Assert(i.Functor(1,2)==6);

    // test ++ --
    Op j = new Op(10);
    j.PlusPlusPrefix();
    j.PlusPlusPostfix(0);
    Assert(j.getI() == 12);
    j.MinusMinusPrefix();
    j.MinusMinusPostfix(0);
    Assert(j.getI() == 10);
    {
      Op op = j.PlusPlusPostfix(0);
      Assert(j.getI() == op.getI()+1);
    }
    {
      Op op = j.MinusMinusPostfix(0);
      Assert(j.getI() == op.getI()-1);
    }

    // cast operators
    Op k = new Op(3);
    int check_k = k.IntCast();
    Assert(check_k == 3);

    Op l = new Op(4);
    double check_l = l.DoubleCast();
    Assert(check_l == 4);

  }

  public static void Assert(boolean b) {
    if (!b)
      throw new RuntimeException("Assertion failed");
  }
}
