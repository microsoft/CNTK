
// This is the java_lib_arrays runtime testcase. It ensures that a getter and a setter has 
// been produced for array members and that they function as expected. It is a
// pretty comprehensive test for all the Java array library typemaps.

import java_lib_arrays.*;

public class java_lib_arrays_runme {

  static {
    try {
	System.loadLibrary("java_lib_arrays");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) {

    // Check array member variables
    ArrayStruct as = new ArrayStruct();

    // Create arrays for all the array types that ArrayStruct can handle
    String array_c = "X";
    byte[] array_c_extra = {11, 22};
    byte[] array_sc = {10, 20};
    short[] array_uc = {101, 201};
    short[] array_s = {1002, 2002};
    int[] array_us = {1003, 2003};
    int[] array_i = {1004, 2004};
    long[] array_ui = {1005, 2005};
    int[] array_l = {1006, 2006};
    long[] array_ul = {1007, 2007};
    long[] array_ll = {1008, 2008};
    float[] array_f = {1009.1f, 2009.1f};
    double[] array_d = {1010.2f, 2010.2f};
    int[] array_finger = {finger.Three, finger.Four};
    int[] array_toe = {toe.Big, toe.Little};

    SimpleStruct[] array_struct={new SimpleStruct(), new SimpleStruct()};
    array_struct[0].setDouble_field(222.333);
    array_struct[1].setDouble_field(444.555);

    AnotherStruct[] array_another_struct={new AnotherStruct(), new AnotherStruct()};
    array_another_struct[0].setSimple(array_struct[0]);
    array_another_struct[1].setSimple(array_struct[1]);

    YetAnotherStruct[] array_yet_another_struct={new YetAnotherStruct(), new YetAnotherStruct()};
    array_yet_another_struct[0].setSimple(array_struct[0]);
    array_yet_another_struct[1].setSimple(array_struct[1]);

    if (array_another_struct[0].getSimple().getDouble_field() != 222.333) throw new RuntimeException("AnotherStruct[0] failed");
    if (array_another_struct[1].getSimple().getDouble_field() != 444.555) throw new RuntimeException("AnotherStruct[1] failed");

    if (java_lib_arrays.extract_ptr(array_yet_another_struct, 0) != 222.333) throw new RuntimeException("extract_ptr 0 failed");
    if (java_lib_arrays.extract_ptr(array_yet_another_struct, 1) != 444.555) throw new RuntimeException("extract_ptr 1 failed");

    java_lib_arrays.modifyYAS(array_yet_another_struct, array_yet_another_struct.length);
    for (int i=0; i<2; ++i) {
      if (array_yet_another_struct[i].getSimple().getDouble_field() != array_struct[i].getDouble_field() * 10.0)
        throw new RuntimeException("modifyYAS failed ");
    }

    java_lib_arrays.toestest(array_toe, array_toe, array_toe);

    // Now set the array members and check that they have been set correctly
    as.setArray_c(array_c);
    check_string(array_c, as.getArray_c());

    as.setArray_sc(array_sc);
    check_byte_array(array_sc, as.getArray_sc());

    as.setArray_uc(array_uc);
    check_short_array(array_uc, as.getArray_uc());

    as.setArray_s(array_s);
    check_short_array(array_s, as.getArray_s());

    as.setArray_us(array_us);
    check_int_array(array_us, as.getArray_us());

    as.setArray_i(array_i);
    check_int_array(array_i, as.getArray_i());

    as.setArray_ui(array_ui);
    check_long_array(array_ui, as.getArray_ui());

    as.setArray_l(array_l);
    check_int_array(array_l, as.getArray_l());

    as.setArray_ul(array_ul);
    check_long_array(array_ul, as.getArray_ul());

    as.setArray_ll(array_ll);
    check_long_array(array_ll, as.getArray_ll());

    as.setArray_f(array_f);
    check_float_array(array_f, as.getArray_f());

    as.setArray_d(array_d);
    check_double_array(array_d, as.getArray_d());

    as.setArray_enum(array_finger);
    check_int_array(array_finger, as.getArray_enum());

    as.setArray_struct(array_struct);
    check_struct_array(array_struct, as.getArray_struct());

    // Extended element (for char[])
    ArrayStructExtra ase = new ArrayStructExtra();
    ase.setArray_c2(array_c_extra);
    check_byte_array(array_c_extra, ase.getArray_c2());

 }

  // Functions to check that the array values were set correctly
  public static void check_string(String original, String checking) {
    if (!checking.equals(original)) {
      throw new RuntimeException("Runtime test failed. checking = [" + checking + "]");
    }
  }
  public static void check_byte_array(byte[] original, byte[] checking) {
    for (int i=0; i<original.length; i++) {
      if (checking[i] != original[i]) {
        throw new RuntimeException("Runtime test failed. checking[" + i + "]=" + checking[i]);
      }
    }
  }
  public static void check_short_array(short[] original, short[] checking) {
    for (int i=0; i<original.length; i++) {
      if (checking[i] != original[i]) {
        throw new RuntimeException("Runtime test failed. checking[" + i + "]=" + checking[i]);
      }
    }
  }
  public static void check_int_array(int[] original, int[] checking) {
    for (int i=0; i<original.length; i++) {
      if (checking[i] != original[i]) {
        throw new RuntimeException("Runtime test failed. checking[" + i + "]=" + checking[i]);
      }
    }
  }
  public static void check_long_array(long[] original, long[] checking) {
    for (int i=0; i<original.length; i++) {
      if (checking[i] != original[i]) {
        throw new RuntimeException("Runtime test failed. checking[" + i + "]=" + checking[i]);
      }
    }
  }
  public static void check_float_array(float[] original, float[] checking) {
    for (int i=0; i<original.length; i++) {
      if (checking[i] != original[i]) {
        throw new RuntimeException("Runtime test failed. checking[" + i + "]=" + checking[i]);
      }
    }
  }
  public static void check_double_array(double[] original, double[] checking) {
    for (int i=0; i<original.length; i++) {
      if (checking[i] != original[i]) {
        throw new RuntimeException("Runtime test failed. checking[" + i + "]=" + checking[i]);
      }
    }
  }
  public static void check_struct_array(SimpleStruct[] original, SimpleStruct[] checking) {
    for (int i=0; i<original.length; i++) {
      if (checking[i].getDouble_field() != original[i].getDouble_field()) {
        throw new RuntimeException("Runtime test failed. checking[" + i + "].double_field=" + checking[i].getDouble_field());
      }
    }
  }
}
