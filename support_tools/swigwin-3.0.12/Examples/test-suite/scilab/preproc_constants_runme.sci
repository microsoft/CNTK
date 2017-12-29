exec("swigtest.start", -1);

checkequal(CONST_INT1_get(), 10, "CONST_INT1");
checkequal(CONST_DOUBLE3_get(), 12.3, "CONST_DOUBLE3");
checkequal(CONST_BOOL1_get(), %T, "CONST_BOOL1");
checkequal(CONST_CHAR_get(), 'x', "CONST_CHAR");
checkequal(CONST_STRING1_get(), "const string", "CONST_STRING1");

// Test global constants can be seen within functions
function test_global()
  global CONST_INT1
  global CONST_DOUBLE3
  global CONST_BOOL1
  global CONST_CHAR
  global CONST_STRING1

  checkequal(CONST_INT1_get(), 10, "CONST_INT1");
  checkequal(CONST_DOUBLE3_get(), 12.3, "CONST_DOUBLE3");
  checkequal(CONST_BOOL1_get(), %T, "CONST_BOOL1");
  checkequal(CONST_CHAR_get(), 'x', "CONST_CHAR");
  checkequal(CONST_STRING1_get(), "const string", "CONST_STRING1");
endfunction

test_global();


// Test assignement in enums
checkequal(kValue_get(), 4, "kValue");

exec("swigtest.quit", -1);
