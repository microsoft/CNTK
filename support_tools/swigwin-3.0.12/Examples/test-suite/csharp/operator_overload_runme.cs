using System;
using System.Threading;
using operator_overloadNamespace;

public class runme
{
  static void Main() 
  {
    Op.sanity_check();
    {
      Op op = new Op(100);
      Op opNew = op++;
      if (op.i != 101) throw new Exception("operator++ postfix failed (op)");
      if (opNew.i != 100) throw new Exception("operator++ postfix failed (opNew)");
    }
    {
      Op op = new Op(100);
      Op opNew = op--;
      if (op.i != 99) throw new Exception("operator-- postfix failed (op)");
      if (opNew.i != 100) throw new Exception("operator-- postfix failed (opNew)");
    }
    {
      Op op = new Op(100);
      Op opNew = ++op;
      if (op.i != 101) throw new Exception("operator++ prefix failed (op)");
      if (opNew.i != 101) throw new Exception("operator++ prefix failed (opNew)");
    }
    {
      Op op = new Op(100);
      Op opNew = --op;
      if (op.i != 99) throw new Exception("operator-- prefix failed (op)");
      if (opNew.i != 99) throw new Exception("operator-- prefix failed (opNew)");
    }

    // overloaded operator class
    Op k = new OpDerived(3);
    int check_k = k.IntCast();
    Assert(check_k == 6);
  }

  public static void Assert(bool b) {
    if (!b)
      throw new Exception("Assertion failed");
  }
}

