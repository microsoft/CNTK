exec("swigtest.start", -1);


// double
checkequal(in_double(22.22), 22.22, "in_double");
checkequal(inr_double(22.22), 22.22, "inr_double");
checkequal(out_double(22.22), 22.22, "out_double");
checkequal(outr_double(22.22), 22.22, "outr_double");
checkequal(inout_double(22.22), 22.22, "inout_double");
checkequal(inoutr_double(22.22), 22.22, "inoutr_double");

// signed char
checkequal(in_schar(22), 22, "in_schar");
checkequal(inr_schar(22), 22, "inr_schar");
checkequal(out_schar(22), 22, "out_schar");
checkequal(outr_schar(22), 22, "outr_schar");
checkequal(inout_schar(22), 22, "inout_schar");
checkequal(inoutr_schar(22), 22, "inoutr_schar");

// unsigned char
checkequal(in_uchar(uint8(22)), 22, "in_uchar");
checkequal(inr_uchar(uint8(22)), 22, "inr_uchar");
checkequal(out_uchar(uint8(22)), 22, "out_uchar");
checkequal(outr_uchar(uint8(22)), 22, "outr_uchar");
checkequal(inout_uchar(uint8(22)), 22, "inout_uchar");
checkequal(inoutr_uchar(uint8(22)), 22, "inoutr_uchar");

// short
checkequal(in_short(22), 22, "in_short");
checkequal(inr_short(22), 22, "inr_short");
checkequal(out_short(22), 22, "out_short");
checkequal(outr_short(22), 22, "outr_short");
checkequal(inout_short(22), 22, "inout_short");
checkequal(inoutr_short(22), 22, "inoutr_short");

// unsigned short
checkequal(in_ushort(uint16(22)), 22, "in_ushort");
checkequal(inr_ushort(uint16(22)), 22, "in_ushort");
checkequal(out_ushort(uint16(22)), 22, "out_ushort");
checkequal(outr_ushort(uint16(22)), 22, "outr_ushort");
checkequal(inout_ushort(uint16(22)), 22, "inout_ushort");
checkequal(inoutr_ushort(uint16(22)), 22, "inoutr_ushort");

// int
checkequal(in_int(22), 22, "in_int");
checkequal(inr_int(22), 22, "inr_int");
checkequal(out_int(22), 22, "out_int");
checkequal(outr_int(22), 22, "outr_int");
checkequal(inout_int(22), 22, "inout_int");
checkequal(inoutr_int(22), 22, "inoutr_int");

// unsigned int
checkequal(in_uint(uint32(22)), 22, "in_uint");
checkequal(inr_uint(uint32(22)), 22, "inr_uint");
checkequal(out_uint(uint32(22)), 22, "out_uint");
checkequal(outr_uint(uint32(22)), 22, "outr_uint");
checkequal(inout_uint(uint32(22)), 22, "inout_uint");
checkequal(inoutr_uint(uint32(22)), 22, "inoutr_uint");

// long
checkequal(in_long(22), 22, "in_long");
checkequal(inr_long(22), 22, "inr_long");
checkequal(out_long(22), 22, "out_long");
checkequal(outr_long(22), 22, "outr_long");
checkequal(inout_long(22), 22, "inout_long");
checkequal(inoutr_long(22), 22, "inoutr_long");

// unsigned long
checkequal(in_ulong(uint32(22)), 22, "in_ulong");
checkequal(inr_ulong(uint32(22)), 22, "inr_ulong");
checkequal(out_ulong(uint32(22)), 22, "out_ulong");
checkequal(outr_ulong(uint32(22)), 22, "outr_ulong");
checkequal(inout_ulong(uint32(22)), 22, "inout_ulong");
checkequal(inoutr_ulong(uint32(22)), 22, "inoutr_ulong");

// bool
checkequal(in_bool(%t), %t, "in_bool");
checkequal(inr_bool(%f), %f, "inr_bool");
checkequal(out_bool(%t), %t, "out_bool");
checkequal(outr_bool(%f), %f, "outr_bool");
checkequal(inout_bool(%t), %t, "inout_bool");
checkequal(inoutr_bool(%f), %f, "inoutr_bool");

// float
checkequal(in_float(2.5), 2.5, "in_float");
checkequal(inr_float(2.5), 2.5, "inr_float");
checkequal(out_float(2.5), 2.5, "out_float");
checkequal(outr_float(2.5), 2.5, "outr_float");
checkequal(inout_float(2.5), 2.5, "inout_float");
checkequal(inoutr_float(2.5), 2.5, "inoutr_float");

// long long
// Not supported in Scilab 5.5
//checkequal(in_longlong(22), 22, "in_longlong");
//checkequal(inr_longlong(22), 22, "inr_longlong");
//checkequal(out_longlong(22), 22, "out_longlong");
//checkequal(outr_longlong(22), 22, "outr_longlong");
//checkequal(inout_longlong(22), 22, "inout_longlong");
//checkequal(inoutr_longlong(22), 22, "inoutr_longlong");

// unsigned long long
// Not supported in Scilab 5.5
//checkequal(in_ulonglong(uint64(22)), 22, "in_ulonglong");
//checkequal(inr_ulonglong(uint64(22)), 22, "inr_ulonglong");
//checkequal(out_ulonglong(uint64(22)), 22, "out_ulonglong");
//checkequal(outr_ulonglong(uint64(22)), 22, "outr_ulonglong");
//checkequal(inout_ulonglong(uint64(22)), 22, "inout_ulonglong");
//checkequal(inoutr_ulonglong(uint64(22)), 22, "inoutr_ulonglong");

// the others
//a,b = inoutr_int2(1, 2);
//checkequal(a<>1 || b<>2), "");
//f,i = out_foo(10)
//checkequal(f.a, 10 || i, 20), "");

exec("swigtest.quit", -1);
