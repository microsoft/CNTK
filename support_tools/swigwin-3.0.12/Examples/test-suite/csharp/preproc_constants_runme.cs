using System;
using System.Reflection;
using preproc_constantsNamespace;

public class runme {
  static void Main() {
    assert( typeof(int) == preproc_constants.CONST_INT1.GetType() );
    assert( typeof(int) == preproc_constants.CONST_INT2.GetType() );
    assert( typeof(uint) == preproc_constants.CONST_UINT1.GetType() );
    assert( typeof(uint) == preproc_constants.CONST_UINT2.GetType() );
    assert( typeof(uint) == preproc_constants.CONST_UINT3.GetType() );
    assert( typeof(uint) == preproc_constants.CONST_UINT4.GetType() );
    assert( typeof(int) == preproc_constants.CONST_LONG1.GetType() );
    assert( typeof(int) == preproc_constants.CONST_LONG2.GetType() );
    assert( typeof(int) == preproc_constants.CONST_LONG3.GetType() );
    assert( typeof(int) == preproc_constants.CONST_LONG4.GetType() );
    assert( typeof(long) == preproc_constants.CONST_LLONG1.GetType() );
    assert( typeof(long) == preproc_constants.CONST_LLONG2.GetType() );
    assert( typeof(long) == preproc_constants.CONST_LLONG3.GetType() );
    assert( typeof(long) == preproc_constants.CONST_LLONG4.GetType() );
    assert( typeof(ulong) == preproc_constants.CONST_ULLONG1.GetType() );
    assert( typeof(ulong) == preproc_constants.CONST_ULLONG2.GetType() );
    assert( typeof(ulong) == preproc_constants.CONST_ULLONG3.GetType() );
    assert( typeof(ulong) == preproc_constants.CONST_ULLONG4.GetType() );
    assert( typeof(double) == preproc_constants.CONST_DOUBLE1.GetType() );
    assert( typeof(double) == preproc_constants.CONST_DOUBLE2.GetType() );
    assert( typeof(double) == preproc_constants.CONST_DOUBLE3.GetType() );
    assert( typeof(double) == preproc_constants.CONST_DOUBLE4.GetType() );
    assert( typeof(double) == preproc_constants.CONST_DOUBLE5.GetType() );
    assert( typeof(double) == preproc_constants.CONST_DOUBLE6.GetType() );
    assert( typeof(bool) == preproc_constants.CONST_BOOL1.GetType() );
    assert( typeof(bool) == preproc_constants.CONST_BOOL2.GetType() );
    assert( typeof(char) == preproc_constants.CONST_CHAR.GetType() );
    assert( typeof(string) == preproc_constants.CONST_STRING1.GetType() );
    assert( typeof(string) == preproc_constants.CONST_STRING2.GetType() );

    assert( typeof(int) == preproc_constants.INT_AND_BOOL.GetType() );
//    assert( typeof(int) == preproc_constants.INT_AND_CHAR.GetType() );
    assert( typeof(int) == preproc_constants.INT_AND_INT.GetType() );
    assert( typeof(uint) == preproc_constants.INT_AND_UINT.GetType() );
    assert( typeof(int) == preproc_constants.INT_AND_LONG.GetType() );
    assert( typeof(uint) == preproc_constants.INT_AND_ULONG.GetType() );
    assert( typeof(long) == preproc_constants.INT_AND_LLONG.GetType() );
    assert( typeof(ulong) == preproc_constants.INT_AND_ULLONG.GetType() );
    assert( typeof(int ) == preproc_constants.BOOL_AND_BOOL.GetType() );

    assert( typeof(int) == preproc_constants.EXPR_MULTIPLY.GetType() );
    assert( typeof(int) == preproc_constants.EXPR_DIVIDE.GetType() );
    assert( typeof(int) == preproc_constants.EXPR_PLUS.GetType() );
    assert( typeof(int) == preproc_constants.EXPR_MINUS.GetType() );
    assert( typeof(int) == preproc_constants.EXPR_LSHIFT.GetType() );
    assert( typeof(int) == preproc_constants.EXPR_RSHIFT.GetType() );
    assert( typeof(bool) == preproc_constants.EXPR_LTE.GetType() );
    assert( typeof(bool) == preproc_constants.EXPR_GTE.GetType() );
    assert( typeof(bool) == preproc_constants.EXPR_INEQUALITY.GetType() );
    assert( typeof(bool) == preproc_constants.EXPR_EQUALITY.GetType() );
    assert( typeof(int) == preproc_constants.EXPR_AND.GetType() );
    assert( typeof(int) == preproc_constants.EXPR_XOR.GetType() );
    assert( typeof(int) == preproc_constants.EXPR_OR.GetType() );
    assert( typeof(bool) == preproc_constants.EXPR_LAND.GetType() );
    assert( typeof(bool) == preproc_constants.EXPR_LOR.GetType() );
    assert( typeof(double) == preproc_constants.EXPR_CONDITIONAL.GetType() );

  }
  static void assert(bool assertion) {
    if (!assertion)
      throw new ApplicationException("test failed");
  }
}
