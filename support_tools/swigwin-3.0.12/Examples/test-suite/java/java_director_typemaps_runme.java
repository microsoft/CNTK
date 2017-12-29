// tests for java/typemaps.i used for directors

import java_director_typemaps.*;
import java.math.BigInteger;

public class java_director_typemaps_runme {

  static {
    try {
      System.loadLibrary("java_director_typemaps");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }


  public static void main(String argv[]) {
    Quux quux = new java_director_typemaps_MyQuux();
    quux.etest();
  }
}

class java_director_typemaps_MyQuux extends Quux {
  public java_director_typemaps_MyQuux() {
    super();
  }

  public void director_method_bool_output(
      boolean[] bool_arg,

      byte[] signed_char_arg,
      short[] unsigned_char_arg,

      short[] short_arg,
      int[] unsigned_short_arg,

      int[] int_arg,
      long[] unsigned_int_arg,

      int[] long_arg,
      long[] unsigned_long_arg,

      long[] long_long_arg,
      // BigInteger[] unsigned_long_long_arg,

      float[] float_arg,
      double[] double_arg)
  {
    bool_arg[0] = true;
    signed_char_arg[0] = 1;
    unsigned_char_arg[0] = 2;
    short_arg[0] = 3;
    unsigned_short_arg[0] = 4;
    int_arg[0] = 5;
    unsigned_int_arg[0] = 6;
    long_arg[0] = 7;
    unsigned_long_arg[0] = 8;
    long_long_arg[0] = 9;
    // unsigned_long_long_arg[0] = 10;
    float_arg[0] = 11;
    double_arg[0] = 12;
  }

  public void director_method_bool_inout(
      boolean[] bool_arg,

      byte[] signed_char_arg,
      short[] unsigned_char_arg,

      short[] short_arg,
      int[] unsigned_short_arg,

      int[] int_arg,
      long[] unsigned_int_arg,

      int[] long_arg,
      long[] unsigned_long_arg,

      long[] long_long_arg,
      // BigInteger[] unsigned_long_long_arg,

      float[] float_arg,
      double[] double_arg)
  {
    if (bool_arg[0]) throw new RuntimeException("unexpected value for bool_arg");

    if (signed_char_arg[0] != 101)  throw new RuntimeException("unexpected value for signed_char_arg");
    if (unsigned_char_arg[0] != 101)  throw new RuntimeException("unexpected value for unsigned_char_arg");
    if (short_arg[0] != 101)  throw new RuntimeException("unexpected value for short_arg");
    if (unsigned_short_arg[0] != 101)  throw new RuntimeException("unexpected value for unsigned_short_arg");
    if (int_arg[0] != 101)  throw new RuntimeException("unexpected value for int_arg");
    if (unsigned_int_arg[0] != 101)  throw new RuntimeException("unexpected value for unsigned_int_arg");
    if (long_arg[0] != 101)  throw new RuntimeException("unexpected value for long_arg");
    if (unsigned_long_arg[0] != 101)  throw new RuntimeException("unexpected value for unsigned_long_arg");
    if (long_long_arg[0] != 101)  throw new RuntimeException("unexpected value for long_long_arg");
    // if (unsigned_long_long_arg[0] != 101)  throw new RuntimeException("unexpected value for unsigned_long_long_arg");
    if (float_arg[0] != 101)  throw new RuntimeException("unexpected value for float_arg");
    if (double_arg[0] != 101)  throw new RuntimeException("unexpected value for double_arg");

    bool_arg[0] = false;
    signed_char_arg[0] = 11;
    unsigned_char_arg[0] = 12;
    short_arg[0] = 13;
    unsigned_short_arg[0] = 14;
    int_arg[0] = 15;
    unsigned_int_arg[0] = 16;
    long_arg[0] = 17;
    unsigned_long_arg[0] = 18;
    long_long_arg[0] = 19;
    // unsigned_long_long_arg[0] = 110;
    float_arg[0] = 111;
    double_arg[0] = 112;
  }

  public void director_method_bool_nameless_args(
      boolean[] bool_arg,

      byte[] signed_char_arg,
      short[] unsigned_char_arg,

      short[] short_arg,
      int[] unsigned_short_arg,

      int[] int_arg,
      long[] unsigned_int_arg,

      int[] long_arg,
      long[] unsigned_long_arg,

      long[] long_long_arg,
      // BigInteger[] unsigned_long_long_arg,

      float[] float_arg,
      double[] double_arg)
  {
    bool_arg[0] = true;
    signed_char_arg[0] = 12;
    unsigned_char_arg[0] = 13;
    short_arg[0] = 14;
    unsigned_short_arg[0] = 15;
    int_arg[0] = 16;
    unsigned_int_arg[0] = 17;
    long_arg[0] = 18;
    unsigned_long_arg[0] = 19;
    long_long_arg[0] = 20;
    // unsigned_long_long_arg[0] = 111;
    float_arg[0] = 112;
    double_arg[0] = 113;
  }
}
