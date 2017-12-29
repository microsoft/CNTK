
using System;
using li_swigtype_inoutNamespace;

public class li_swigtype_inout_runme {

  public static void Main() {
    XXX xxx = new XXX(999);
    check_count(1);
    XXX x1 = null;
    XXX x2 = null;
    XXX x3 = null;
    XXX x4 = null;
    li_swigtype_inout.ptr_ref_out(out x1, out x2, out x3, out x4);
    check_value(111, x1.value);
    check_value(222, x2.value);
    check_value(333, x3.value);
    check_value(444, x4.value);
    check_count(5);
    x1.Dispose();
    x2.Dispose();
    x3.Dispose();
    x4.Dispose();
    xxx.Dispose();
    check_count(0);

    x1 = null;
    x2 = null;
    x3 = null;
    x4 = null;
    new ConstructorTest(out x1, out x2, out x3, out x4);
    check_count(4);
    check_value(111, x1.value);
    check_value(222, x2.value);
    check_value(333, x3.value);
    check_value(444, x4.value);
    x1.Dispose();
    x2.Dispose();
    x3.Dispose();
    x4.Dispose();
    check_count(0);
  }

  public static void check_count(int count) {
    int actual = XXX.count;
      if( count != actual ) {
        throw new Exception(String.Format("Count wrong. Expected: {0} Got: {1}", count, actual));
      }
  }

  public static void check_value(int expected, int actual) {
      if( expected != actual ) {
        throw new Exception(String.Format("Wrong value. Expected: {0} Got: {1}", expected, actual));
      }
  }
}
