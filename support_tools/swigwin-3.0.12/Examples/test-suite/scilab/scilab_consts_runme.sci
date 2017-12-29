exec("swigtest.start", -1);

checkequal(ICONST0_get(), 42, "ICONST0_get()");
checkequal(FCONST0_get(), 2.1828, "FCONST0_get()");
checkequal(CCONST0_get(), "x", "CCONST0_get()");
//checkequal(CCONST0_2_get(), "\n", "CCONST0_2_get()");
checkequal(SCONST0_get(), "Hello World", "SCONST0_get()");
checkequal(SCONST0_2_get(), """Hello World""", "SCONST0_2_get()");
checkequal(EXPR0_get(), 48.5484, "EXPR0_get()");
checkequal(iconst0_get(), 37, "iconst0_get()");
checkequal(fconst0_get(), 42.2, "fconst0_get()");

checkequal(UNSIGNED0_get(), hex2dec("5FFF"), "UNSIGNED0_get()");
checkequal(LONG0_get(), hex2dec("3FFF0000"), "LONG0_get()");
checkequal(ULONG0_get(), hex2dec("5FF0000"), "ULONG0_get()");

if isdef('BAR0') then swigtesterror("BAR0"); end

checkequal(ICONST1, int32(42), "ICONST1");
checkequal(FCONST1, 2.1828, "FCONST1");
checkequal(CCONST1, "x", "CCONST1");
//checkequal(CCONST1_2, "\n", "CCONST1_2");
checkequal(SCONST1, "Hello World", "SCONST1");
checkequal(SCONST1_2, """Hello World""", "SCONST1_2");
checkequal(EXPR1, 48.5484, "EXPR1");
checkequal(iconst1, int32(37), "iconst1");
checkequal(fconst1, 42.2, "fconst1");

checkequal(UNSIGNED1, uint32(hex2dec("5FFF")), "UNSIGNED1");
checkequal(LONG1, int32(hex2dec("3FFF0000")), "LONG1");
checkequal(ULONG1, uint32(hex2dec("5FF0000")), "ULONG1");

if isdef('BAR1') then swigtesterror("BAR1"); end

exec("swigtest.quit", -1);
