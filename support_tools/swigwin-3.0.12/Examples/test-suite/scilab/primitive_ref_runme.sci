exec("swigtest.start", -1);

checkequal(ref_int(3), 3, "ref_int() test fails.");
checkequal(ref_uint(uint32(3)), 3, "ref_uint() test fails.");

checkequal(ref_short(3), 3, "ref_short() test fails.");
checkequal(ref_ushort(uint16(3)), 3, "ref_ushort() test fails.");

checkequal(ref_long(3), 3, "ref_long() test fails.");
checkequal(ref_ulong(uint32(3)), 3, "ref_ulong() test fails.");

checkequal(ref_schar(3), 3, "ref_schar() test fails.");
checkequal(ref_uchar(uint8(3)), 3, "ref_uchar() test fails.");

checkequal(ref_float(3), 3, "ref_float() test fails.");
checkequal(ref_double(3), 3, "ref_double() test fails.");

checkequal(ref_bool(%T), %T, "ref_bool() test fails.");

checkequal(ref_char('x'), 'x', "ref_char() test fails.");

checkequal(ref_over(0), 0, "ref_over() test fails.");

exec("swigtest.quit", -1);
