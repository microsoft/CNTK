exec("swigtest.start", -1);

function checkEnum(enum_val, expected_enum_val)
  if typeof(enum_val) <> "constant" then swigtesterror(); end
  if enum_val <> expected_enum_val then swigtesterror(); end
endfunction

checkEnum(ENUM_1, 0);
checkEnum(ENUM_2, 1);

checkEnum(ENUM_EXPLICIT_1_1, 5);
checkEnum(ENUM_EXPLICIT_1_2, 6);

checkEnum(ENUM_EXPLICIT_2_1, 0);
checkEnum(ENUM_EXPLICIT_2_2, 10);

checkEnum(ENUM_EXPLICIT_3_1, 2);
checkEnum(ENUM_EXPLICIT_3_2, 5);
checkEnum(ENUM_EXPLICIT_3_3, 8);

checkEnum(TYPEDEF_ENUM_1_1, 21);
checkEnum(TYPEDEF_ENUM_1_2, 22);

checkEnum(TYPEDEF_ENUM_2_1, 31);
checkEnum(TYPEDEF_ENUM_2_2, 32);

checkEnum(ENUM_REF_1, 1);
checkEnum(ENUM_REF_2, 10);

checkEnum(clsEnum_CLS_ENUM_1, 100);
checkEnum(clsEnum_CLS_ENUM_2, 101);

checkEnum(clsEnum_CLS_ENUM_REF_1, 101);
checkEnum(clsEnum_CLS_ENUM_REF_2, 110);

exec("swigtest.quit", -1);
