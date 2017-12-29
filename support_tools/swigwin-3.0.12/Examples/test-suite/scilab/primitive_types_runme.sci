exec("swigtest.start", -1);

// Check passing by value

checkequal(val_double(42), 42, "val_double() test fails.");
checkequal(val_float(42), 42, "val_float() test fails.");

checkequal(val_char('a'), 'a', "val_char() test fails.");
checkequal(val_schar(42), 42, "val_schar() test fails.");
checkequal(val_schar(int8(42)), 42, "val_schar() test fails.");
checkequal(val_uchar(uint8(42)), 42, "val_uchar() test fails.");

checkequal(val_short(42), 42, "val_short() test fails.");
checkequal(val_short(int16(42)), 42, "val_short() test fails.");
checkequal(val_ushort(uint16(42)), 42, "val_ushort() test fails.");

checkequal(val_int(42), 42, "val_int() test fails.");
checkequal(val_int(int32(42)), 42, "val_int() test fails.");
checkequal(val_uint(uint32(42)), 42, "val_uint() test fails.");

checkequal(val_long(42), 42, "val_long() test fails.");
checkequal(val_long(int32(42)), 42, "val_long() test fails.");
checkequal(val_ulong(uint32(42)), 42, "val_long() test fails.");

checkequal(val_bool(%t), %t, "val_bool() test fails.");

// longlong is not supported in Scilab 5.x
//checkequal(val_llong(42), 42, "val_llong() test fails.");
//checkequal(val_llong(int64(42)), 42, "val_llong() test fails.");
//checkequal(val_ullong(uint64(42)), 42, "val_ullong() test fails.");

// Check passing by reference
checkequal(ref_char('a'), 'a', "ref_char() test fails.");
checkequal(ref_schar(42), 42, "ref_schar() test fails.");
checkequal(ref_schar(int8(42)), 42, "ref_schar() test fails.");
checkequal(ref_uchar(uint8(42)), 42, "ref_uchar() test fails.");

checkequal(ref_short(42), 42, "ref_short() test fails.")
checkequal(ref_short(int16(42)), 42, "ref_short() test fails.")
checkequal(ref_ushort(uint16(42)), 42, "ref_ushort() test fails.")

checkequal(ref_int(42), 42, "ref_int() test fails.");
checkequal(ref_int(int32(42)), 42, "ref_int() test fails.");
checkequal(ref_uint(uint32(42)), 42, "ref_uint() test fails.");

checkequal(ref_long(42), 42, "ref_long() test fails.");
checkequal(ref_long(int32(42)), 42, "ref_long() test fails.");
checkequal(ref_ulong(uint32(42)), 42, "ref_ulong() test fails.");

checkequal(ref_bool(%t), %t, "ref_bool() test fails.");

// long long is not supported in Scilab 5.x
//checkequal(ref_llong(42), 42, "ref_llong() test fails.");
//checkequal(ref_llong(int64(42)), 42, "ref_llong() test fails.");
//checkequal(ref_ullong(uint64(42)), 42, "ref_ullong() test fails.");

exec("swigtest.quit", -1);
