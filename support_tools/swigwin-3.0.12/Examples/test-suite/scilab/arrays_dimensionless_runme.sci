exec("swigtest.start", -1);

// bool
checkequal(arr_bool([%T %F %F %T %F %T %T %T], 8), 5, "arr_bool");

// char
checkequal(arr_char(["charptr"], 7), 756, "arr_char");

// signed char
checkequal(arr_schar([1, 2, 3, 4], 4), 10, "arr_schar");
checkequal(arr_schar(int8([1, 2, 3, 4]), 4), 10, "arr_schar");

// unsigned char
checkequal(arr_uchar([1, 2, 3, 4], 4), 10, "arr_uchar");
checkequal(arr_uchar(uint8([1, 2, 3, 4]), 4), 10, "arr_uchar");

// short
checkequal(arr_short([1, 2, 3, 4], 4), 10, "arr_short");
checkequal(arr_short(int16([1, 2, 3, 4]), 4), 10, "arr_short");

// unsigned short
checkequal(arr_ushort([1, 2, 3, 4], 4), 10, "arr_ushort");
checkequal(arr_ushort(uint16([1, 2, 3, 4]), 4), 10, "arr_ushort");

// int
checkequal(arr_int([1, 2, 3, 4], 4), 10, "arr_int");
checkequal(arr_int(int32([1, 2, 3, 4]), 4), 10, "arr_int");

// unsigned int
checkequal(arr_uint([1, 2, 3, 4], 4), 10, "");
checkequal(arr_uint(uint32([1, 2, 3, 4]), 4), 10, "");

// long
checkequal(arr_long([1, 2, 3, 4], 4), 10, "arr_long");
checkequal(arr_long(int32([1, 2, 3, 4]), 4), 10, "arr_long");

// unsigned long
checkequal(arr_ulong([1, 2, 3, 4], 4), 10, "arr_ulong");
checkequal(arr_ulong(uint32([1, 2, 3, 4]), 4), 10, "arr_ulong");

// long long
// No equivalent in Scilab 5

// unsigned long long
// No equivalent in Scilab 5

// float
a = [1, 2, 3, 4];
checkequal(arr_float(a, 4), 10, "arr_float");

// double
a = [1, 2, 3, 4];
checkequal(arr_double(a, 4), 10, "arr_double");

exec("swigtest.quit", -1);
