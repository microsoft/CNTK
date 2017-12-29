exec("swigtest.start", -1);

x = "hello";

// li_std_string tests

// Function tests

checkequal(test_ccvalue(x), x, "test_ccvalue()");
checkequal(test_cvalue(x), x,  "test_cvalue(x)");
checkequal(test_value(x), x, "test_value()");

checkequal(test_const_reference(x), x, "test_const_reference(x)");
checkequal(test_reference_input(x), x, "test_reference_input(x)");
checkequal(test_reference_inout(x), x+x, "test_reference_inout(x)");

//checkequal(test_reference_out(), "test_reference_out message", "test_reference_out()");
//checkequal(test_const_pointer_out(), "x", "test_const_pointer_out()");

s = "initial string";

// Global variable tests

checkequal(GlobalString2_get(), "global string 2", "GlobalString2_get()");
GlobalString2_set(s);
checkequal(GlobalString2_get(), s, "GlobalString2_get()");

checkequal(ConstGlobalString_get(), "const global string", "ConstGlobalString_get()");

// Member variable tests

myStructure = new_Structure();
checkequal(Structure_Str2_get(myStructure), "member string 2", "Structure_Str2_get(myStructure)");

Structure_Str2_set(myStructure, s);
checkequal(Structure_Str2_get(myStructure), s, "Structure_Str2_get(myStructure)");

checkequal(Structure_ConstStr_get(myStructure), "const member string", "Structure_ConstStr_get(myStructure)");

checkequal(Structure_StaticStr2_get(), "static member string 2", "Structure_StaticStr2_get()");

Structure_StaticStr2_set(s);
checkequal(Structure_StaticStr2_get(), s, "Structure_StaticStr2_get()");

checkequal(Structure_ConstStati_get(), "const static member string", "Structure_ConstStaticStr_get()");


checkequal(stdstring_empty(), "", "stdstring_empty()");
checkequal(c_empty(), "", "c_empty()");


// li_std_string_extra tests

//checkequal(test_value_basic1(x), x, "");
//checkequal(test_value_basic2(x), x, "");
//checkequal(test_value_basic3(x), x, "");

exec("swigtest.quit", -1);
