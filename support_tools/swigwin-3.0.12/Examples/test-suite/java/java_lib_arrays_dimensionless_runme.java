
import java_lib_arrays_dimensionless.*;

public class java_lib_arrays_dimensionless_runme {

  static {
    try {
	System.loadLibrary("java_lib_arrays_dimensionless");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[]) {

    boolean bools[] = {true, false, true, false, true};
    byte schars[] = {5, 10, 15, 20, 25};
    short uchars[] = {5, 10, 15, 20, 25};
    short shorts[] = {5, 10, 15, 20, 25};
    int ushorts[] = {5, 10, 15, 20, 25};
    int ints[] = {5, 10, 15, 20, 25};
    long uints[] = {5, 10, 15, 20, 25};
    int longs[] = {5, 10, 15, 20, 25};
    long ulongs[] = {5, 10, 15, 20, 25};
    long lls[] = {5, 10, 15, 20, 25};
    //java.math.BigInteger ulls[] = {5, 10, 15, 20, 25};
    float floats[] = {5, 10, 15, 20, 25};
    double doubles[] = {5, 10, 15, 20, 25};

    if (java_lib_arrays_dimensionless.arr_bool(bools, bools.length) != 3) throw new RuntimeException("test bools 1 failed");
    if (java_lib_arrays_dimensionless.ptr_bool(bools, bools.length) != 2) throw new RuntimeException("test bools 2 failed");
//    if (java_lib_arrays_dimensionless.arr_char(chars, chars.length) != 75) throw new RuntimeException("test chars 1 failed");
//    if (java_lib_arrays_dimensionless.ptr_char(chars, chars.length) != 150) throw new RuntimeException("test chars 2 failed");
    if (java_lib_arrays_dimensionless.arr_schar(schars, schars.length) != 75) throw new RuntimeException("test schars 1 failed");
    if (java_lib_arrays_dimensionless.ptr_schar(schars, schars.length) != 150) throw new RuntimeException("test schars 2 failed");
    if (java_lib_arrays_dimensionless.arr_uchar(uchars, uchars.length) != 75) throw new RuntimeException("test uchars 1 failed");
    if (java_lib_arrays_dimensionless.ptr_uchar(uchars, uchars.length) != 150) throw new RuntimeException("test uchars 2 failed");
    if (java_lib_arrays_dimensionless.arr_short(shorts, shorts.length) != 75) throw new RuntimeException("test shorts 1 failed");
    if (java_lib_arrays_dimensionless.ptr_short(shorts, shorts.length) != 150) throw new RuntimeException("test shorts 2 failed");
    if (java_lib_arrays_dimensionless.arr_ushort(ushorts, ushorts.length) != 75) throw new RuntimeException("test ushorts 1 failed");
    if (java_lib_arrays_dimensionless.ptr_ushort(ushorts, ushorts.length) != 150) throw new RuntimeException("test ushorts 2 failed");
    if (java_lib_arrays_dimensionless.arr_int(ints, ints.length) != 75) throw new RuntimeException("test ints 1 failed");
    if (java_lib_arrays_dimensionless.ptr_int(ints, ints.length) != 150) throw new RuntimeException("test ints 2 failed");
    if (java_lib_arrays_dimensionless.arr_uint(uints, uints.length) != 75) throw new RuntimeException("test uints 1 failed");
    if (java_lib_arrays_dimensionless.ptr_uint(uints, uints.length) != 150) throw new RuntimeException("test uints 2 failed");
    if (java_lib_arrays_dimensionless.arr_long(longs, longs.length) != 75) throw new RuntimeException("test longs 1 failed");
    if (java_lib_arrays_dimensionless.ptr_long(longs, longs.length) != 150) throw new RuntimeException("test longs 2 failed");
    if (java_lib_arrays_dimensionless.arr_ulong(ulongs, ulongs.length) != 75) throw new RuntimeException("test ulongs 1 failed");
    if (java_lib_arrays_dimensionless.ptr_ulong(ulongs, ulongs.length) != 150) throw new RuntimeException("test ulongs 2 failed");
//    if (java_lib_arrays_dimensionless.arr_ll(lls, lls.length) != 75) throw new RuntimeException("test lls 1 failed");
//    if (java_lib_arrays_dimensionless.ptr_ll(lls, lls.length) != 150) throw new RuntimeException("test lls 2 failed");
//    if (java_lib_arrays_dimensionless.arr_ull(ulls, ulls.length) != 75) throw new RuntimeException("test ulls 1 failed");
//    if (java_lib_arrays_dimensionless.ptr_ull(ulls, ulls.length) != 150) throw new RuntimeException("test ulls 2 failed");
    if (java_lib_arrays_dimensionless.arr_float(floats, floats.length) != 75) throw new RuntimeException("test floats 1 failed");
    if (java_lib_arrays_dimensionless.ptr_float(floats, floats.length) != 150) throw new RuntimeException("test floats 2 failed");
    if (java_lib_arrays_dimensionless.arr_double(doubles, doubles.length) != 75) throw new RuntimeException("test doubles 1 failed");
    if (java_lib_arrays_dimensionless.ptr_double(doubles, doubles.length) != 150) throw new RuntimeException("test doubles 2 failed");

  }

}
