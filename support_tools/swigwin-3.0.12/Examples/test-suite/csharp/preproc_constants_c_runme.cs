using System;
using System.Reflection;
using preproc_constants_cNamespace;

// Same as preproc_constants_c.i testcase, but bool types are int instead
public class runme {
  static void Main() {
    assert( typeof(int) == preproc_constants_c.CONST_INT1.GetType() );
    assert( typeof(int) == preproc_constants_c.CONST_INT2.GetType() );
    assert( typeof(uint) == preproc_constants_c.CONST_UINT1.GetType() );
    assert( typeof(uint) == preproc_constants_c.CONST_UINT2.GetType() );
    assert( typeof(uint) == preproc_constants_c.CONST_UINT3.GetType() );
    assert( typeof(uint) == preproc_constants_c.CONST_UINT4.GetType() );
    assert( typeof(int) == preproc_constants_c.CONST_LONG1.GetType() );
    assert( typeof(int) == preproc_constants_c.CONST_LONG2.GetType() );
    assert( typeof(int) == preproc_constants_c.CONST_LONG3.GetType() );
    assert( typeof(int) == preproc_constants_c.CONST_LONG4.GetType() );
    assert( typeof(long) == preproc_constants_c.CONST_LLONG1.GetType() );
    assert( typeof(long) == preproc_constants_c.CONST_LLONG2.GetType() );
    assert( typeof(long) == preproc_constants_c.CONST_LLONG3.GetType() );
    assert( typeof(long) == preproc_constants_c.CONST_LLONG4.GetType() );
    assert( typeof(ulong) == preproc_constants_c.CONST_ULLONG1.GetType() );
    assert( typeof(ulong) == preproc_constants_c.CONST_ULLONG2.GetType() );
    assert( typeof(ulong) == preproc_constants_c.CONST_ULLONG3.GetType() );
    assert( typeof(ulong) == preproc_constants_c.CONST_ULLONG4.GetType() );
    assert( typeof(double) == preproc_constants_c.CONST_DOUBLE1.GetType() );
    assert( typeof(double) == preproc_constants_c.CONST_DOUBLE2.GetType() );
    assert( typeof(double) == preproc_constants_c.CONST_DOUBLE3.GetType() );
    assert( typeof(double) == preproc_constants_c.CONST_DOUBLE4.GetType() );
    assert( typeof(double) == preproc_constants_c.CONST_DOUBLE5.GetType() );
    assert( typeof(double) == preproc_constants_c.CONST_DOUBLE6.GetType() );
    assert( typeof(int) == preproc_constants_c.CONST_BOOL1.GetType() );
    assert( typeof(int) == preproc_constants_c.CONST_BOOL2.GetType() );
    assert( typeof(char) == preproc_constants_c.CONST_CHAR.GetType() );
    assert( typeof(string) == preproc_constants_c.CONST_STRING1.GetType() );
    assert( typeof(string) == preproc_constants_c.CONST_STRING2.GetType() );

    assert( typeof(int) == preproc_constants_c.INT_AND_BOOL.GetType() );
//    assert( typeof(int) == preproc_constants_c.INT_AND_CHAR.GetType() );
    assert( typeof(int) == preproc_constants_c.INT_AND_INT.GetType() );
    assert( typeof(uint) == preproc_constants_c.INT_AND_UINT.GetType() );
    assert( typeof(int) == preproc_constants_c.INT_AND_LONG.GetType() );
    assert( typeof(uint) == preproc_constants_c.INT_AND_ULONG.GetType() );
    assert( typeof(long) == preproc_constants_c.INT_AND_LLONG.GetType() );
    assert( typeof(ulong) == preproc_constants_c.INT_AND_ULLONG.GetType() );
    assert( typeof(int ) == preproc_constants_c.BOOL_AND_BOOL.GetType() );

    assert( typeof(int) == preproc_constants_c.EXPR_MULTIPLY.GetType() );
    assert( typeof(int) == preproc_constants_c.EXPR_DIVIDE.GetType() );
    assert( typeof(int) == preproc_constants_c.EXPR_PLUS.GetType() );
    assert( typeof(int) == preproc_constants_c.EXPR_MINUS.GetType() );
    assert( typeof(int) == preproc_constants_c.EXPR_LSHIFT.GetType() );
    assert( typeof(int) == preproc_constants_c.EXPR_RSHIFT.GetType() );
    assert( typeof(int) == preproc_constants_c.EXPR_LTE.GetType() );
    assert( typeof(int) == preproc_constants_c.EXPR_GTE.GetType() );
    assert( typeof(int) == preproc_constants_c.EXPR_INEQUALITY.GetType() );
    assert( typeof(int) == preproc_constants_c.EXPR_EQUALITY.GetType() );
    assert( typeof(int) == preproc_constants_c.EXPR_AND.GetType() );
    assert( typeof(int) == preproc_constants_c.EXPR_XOR.GetType() );
    assert( typeof(int) == preproc_constants_c.EXPR_OR.GetType() );
    assert( typeof(int) == preproc_constants_c.EXPR_LAND.GetType() );
    assert( typeof(int) == preproc_constants_c.EXPR_LOR.GetType() );
    assert( typeof(double) == preproc_constants_c.EXPR_CONDITIONAL.GetType() );

  }
  static void assert(bool assertion) {
    if (!assertion)
      throw new ApplicationException("test failed");
  }
}
